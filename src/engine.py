import json
import time
import numpy as np
import networkx as nx
import faiss
from sentence_transformers import SentenceTransformer
from collections import Counter
from typing import List, Dict, Any # Added for clarity

class ThesisEngine:
    def __init__(self):
        self.d = 384
        self.graph = nx.Graph()
        
        # 1. Exact Index (Baseline)
        self.vector_index = faiss.IndexFlatL2(self.d)
        
        # 2. HNSW Index (The Challenger)
        # HNSW64 = Graph with 64 links per node (Standard High Accuracy)
        self.hnsw_index = faiss.IndexHNSWFlat(self.d, 64) 
        self.hnsw_index.hnsw.efConstruction = 80  # Build quality
        self.hnsw_index.hnsw.efSearch = 64        # Search quality
        
        self.vectors = None 
        
        # --- UNIFIED INDEX ---
        # Maps "Value" (Brand OR Category) -> List of IDs
        self.inverted_index = {}      
        self.value_counts = {}      
        self.asin_map = {}          
        
        # Cost Coefficients
        self.cost_per_filter_item = 0.000003 
        self.cost_index_batch = 0.020         
        
        print("   [System] Loading embedding model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_data(self, filepath: str, limit: int = None):
        """
        Loads Data. Indexes BOTH 'brand' and 'main_cat' for flexible testing.
        """
        ITEM_ID_KEY = 'asin'
        TITLE_KEY = 'title'
        # We will look for both of these manually
        KEY_BRAND = 'brand'
        KEY_CAT = 'main_cat'
        EDGE_KEY = 'also_buy'   
        
        print(f"   [System] Reading dataset from {filepath}...")
        titles = []
        item_ids = []   
        raw_edges = [] 
        
        # Temporary lists to store metadata before building graph
        meta_brands = []
        meta_cats = []
        
        count = 0
        loaded_count = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    pid = data.get(ITEM_ID_KEY)
                    title = data.get(TITLE_KEY)
                    
                    # Validation: Needs ID and Title
                    if pid and title:
                        item_ids.append(pid)
                        titles.append(title)
                        
                        # Store metadata for indexing
                        meta_brands.append(data.get(KEY_BRAND, "Unknown"))
                        meta_cats.append(data.get(KEY_CAT, "Unknown"))
                        
                        # Edge Collection
                        targets = data.get(EDGE_KEY, [])
                        if isinstance(targets, dict):
                            flat = []
                            for v in targets.values(): flat.extend(v)
                            targets = flat
                        if targets and isinstance(targets, list):
                            raw_edges.append((loaded_count, targets))
                        
                        loaded_count += 1
                        
                    count += 1
                    if limit and loaded_count >= limit: break
                    if count % 50000 == 0:
                        print(f"            ...processed {count} lines (loaded {loaded_count})...")
                            
                except json.JSONDecodeError: continue

        # --- DIAGNOSTICS ---
        print(f"   [DEBUG] Successfully loaded {loaded_count} items.")

        if loaded_count == 0:
             return 

        # --- VECTORIZATION ---
        print(f"   [System] Vectorizing {loaded_count} titles...")
        batch_size = 5000
        all_embeddings = []
        for start in range(0, len(titles), batch_size):
            end = min(start + batch_size, len(titles))
            batch_emb = self.model.encode(titles[start:end], show_progress_bar=False)
            all_embeddings.append(batch_emb)
            
        self.vectors = np.vstack(all_embeddings).astype('float32')
        faiss.normalize_L2(self.vectors)
        
        # Add to Flat Index
        self.vector_index.add(self.vectors)
        
        # Add to HNSW Index (This takes longer to build!)
        print("   [System] Building HNSW Graph Index (This may take a moment)...")
        self.hnsw_index.add(self.vectors)

        # --- GRAPH & UNIFIED INDEX BUILDING ---
        print("   [System] Building Unified Index (Brands + Categories)...")
        self.graph.clear()
        self.inverted_index = {} # Clear
        self.value_counts = {}
        self.asin_map = {}
        
        for i, pid in enumerate(item_ids):
            b = meta_brands[i]
            c = meta_cats[i]
            
            # Add to NetworkX (store both for reference)
            self.graph.add_node(i, asin=pid, brand=b, category=c, title=titles[i])
            self.asin_map[pid] = i
            
            # --- INDEXING MAGIC: Add ID to BOTH lists ---
            
            # 1. Index Brand
            if b:
                if b not in self.inverted_index: self.inverted_index[b] = []
                self.inverted_index[b].append(i)
                self.value_counts[b] = self.value_counts.get(b, 0) + 1
            
            # 2. Index Category (Avoid duplicates if Brand == Category, rare)
            if c and c != b:
                if c not in self.inverted_index: self.inverted_index[c] = []
                self.inverted_index[c].append(i)
                self.value_counts[c] = self.value_counts.get(c, 0) + 1

        # --- EDGE LINKING ---
        print("   [System] Linking Graph Edges...")
        for source_id, target_pids in raw_edges:
            for t_pid in target_pids:
                if t_pid in self.asin_map:
                    target_id = self.asin_map[t_pid]
                    self.graph.add_edge(source_id, target_id, edge_type="RELATED") 
        
        print(f"   [System] Ready. Indexed {len(self.inverted_index)} unique attributes.")
    def calibrate(self):
        """
        SELF-CALIBRATION: 
        Measures lookup speed AND vector math speed to find the true bottleneck.
        """
        if self.vectors is None or len(self.vectors) < 1000:
            print("   [Calibration] Insufficient data. Using defaults.")
            return

        print("   [System] Calibrating Cost Optimizer (Measuring CPU/Memory)...")
        
        # 1. Warm-Up (Crucial to avoid 'cold start' spikes)
        dummy_q = np.random.randn(1, self.d).astype('float32')
        self.vector_index.search(dummy_q, 10)
        
        # --- 2. Measure Pre-Filter Cost (Lookup + Math) ---
        sample_size = 5000
        start = time.perf_counter()
        
        # A. Simulate attribute lookup (The Dictionary part)
        target_val = "NonExistent"
        # We access graph nodes just like the real search does
        for i in range(min(sample_size, len(self.graph))):
            _ = self.graph.nodes[i].get('brand') == target_val
            
        # B. Simulate Vector Math (The Calculation part) - NEW & CRITICAL
        # We calculate distances for 'sample_size' vectors
        if len(self.vectors) >= sample_size:
            sample_vecs = self.vectors[:sample_size]
            diff = sample_vecs - dummy_q
            # This is the heavy math operation: Sum of Squared Differences
            dists = np.sum(diff**2, axis=1)
        
        duration = time.perf_counter() - start
        
        # This now represents the FULL cost of processing one item
        self.cost_per_filter_item = duration / sample_size
        
        # --- 3. Measure Post-Filter Cost (Index Search) ---
        start = time.perf_counter()
        self.vector_index.search(dummy_q, 500)
        self.cost_index_batch = time.perf_counter() - start
        
        # --- 4. Report ---
        print(f"            -> Learned Cost Per Item: {self.cost_per_filter_item*1000:.6f} ms")
        print(f"            -> Learned Index Cost:    {self.cost_index_batch*1000:.4f} ms")
        

        if self.cost_per_filter_item > 0:
            threshold = int(self.cost_index_batch / self.cost_per_filter_item)
            print(f"            -> Dynamic Switch Point:  ~{threshold} items")




    def get_details(self, node_ids: List[int]) -> List[str]:
        """Helper to fetch human-readable details for a list of IDs"""
        details = []
        for nid in node_ids:
            data = self.graph.nodes[nid]
            # Changed 'brand' to 'attribute'
            details.append(f"[{data.get('attribute', 'N/A')}] {data['title'][:80]}...") 
        return details

    def get_contextual_brands(self, query_vec, search_k=5000):
        """Analyzes vector neighborhood to find appropriate test attributes (Author/Publisher)"""
        D, I = self.vector_index.search(query_vec.reshape(1, -1), search_k)
        found_attributes = []
        for nid in I[0]:
            if nid == -1: continue
            # Changed 'brand' to 'attribute'
            attribute_name = self.graph.nodes[nid].get('attribute') 
            if attribute_name: found_attributes.append(attribute_name)
        c = Counter(found_attributes)
        
        def pick(min_c, max_c):
            opts = [b for b, cnt in c.items() if min_c <= cnt <= max_c]
            return (opts[0], c[opts[0]]) if opts else (None, 0)

        # Labels remain the same for generic testing categories
        return {
            "Common": pick(200, 5000),
            "Mid-Tier": pick(20, 50),
            "Rare": pick(2, 5)
        }

    # =========================================
    # TYPE 1: ATTRIBUTE FILTERING (Author)
    # =========================================
    
    def search_pre_filter(self, query_vec, target_attribute, k=10):
        """
        PRE-FILTER STRATEGY:
        1. Look up all IDs for the attribute (Brand).
        2. Fetch their vectors.
        3. Calculate distances manually (Brute Force on subset).
        4. Return top-k.
        """
        start = time.perf_counter()
        
        # 1. Identify Candidates
        allow_list = self.inverted_index.get(target_attribute, [])
        
        # If no items exist for this brand, return empty immediately
        if not allow_list:
            return [], time.perf_counter() - start
            
        # 2. Retrieve Vectors for Candidates
        # (allow_list is a list of integers, so we can slice the numpy array)
        candidate_vectors = self.vectors[allow_list]
        
        # 3. Calculate Distances (L2 Metric to match FAISS)
        # dist = ||a - b||^2
        # Since we want accurate ranking, we calculate full L2.
        # (candidate_vectors - query_vec) gives (N, d) difference
        diff = candidate_vectors - query_vec.reshape(1, -1)
        dists = np.sum(diff**2, axis=1) # Sum of squared differences
        
        # 4. Sort and Pick Top-K
        # We want the *indices* of the smallest distances
        # argsort gives us indices relative to 'candidate_vectors' (0 to N)
        # We need to map them back to global IDs using 'allow_list'
        
        # Optimization: If N is huge, use argpartition instead of full sort
        if len(dists) > k:
            top_local_indices = np.argpartition(dists, k)[:k]
            # Now sort just these top k for perfect ordering
            top_local_indices = top_local_indices[np.argsort(dists[top_local_indices])]
        else:
            top_local_indices = np.argsort(dists)
            
        # Map back to global IDs
        found_ids = [allow_list[i] for i in top_local_indices]
        
        return found_ids, time.perf_counter() - start

    def search_post_filter(self, query_vec, target_attribute, k=10):
        start = time.perf_counter()
        oversample = 50
        D, I = self.vector_index.search(query_vec.reshape(1, -1), k * oversample)
        found_ids = []
        for nid in I[0]:
            if nid == -1: continue
            # Changed 'brand' lookup
            if self.graph.nodes[nid].get('attribute') == target_attribute: 
                found_ids.append(nid)
                if len(found_ids) >= k: break
        # Note: The output tuple must be updated below if you want to track the method used.
        # This function returns (found_ids, elapsed_time)
        return found_ids, time.perf_counter() - start

    def search_adaptive(self, query_vec, target_attribute, k=10):
        """
        True Cost-Based Adaptive Search
        Decides strategy based on the math: (N * Cost_per_item) vs (Index_Cost)
        """
        count = self.value_counts.get(target_attribute, 0)
        
        # A. Estimate Pre-Filter Cost (Linear)
        estimated_pre_cost = count * self.cost_per_filter_item
        
        # B. Estimate Post-Filter Cost (Constant-ish)
        estimated_post_cost = self.cost_index_batch
        
        # C. Compare
        if estimated_pre_cost < estimated_post_cost:
            return self.search_pre_filter(query_vec, target_attribute, k) + ("PRE-FILTER",)
        else:
            return self.search_post_filter(query_vec, target_attribute, k) + ("POST-FILTER",)

    def search_post_filter_hnsw(self, query_vec, target_attribute, k=10):
        """
        HNSW Post-Filter: 
        Fast Graph Search -> Filter Results.
        """
        start = time.perf_counter()
        
        # HNSW needs a bigger initial k because it is approximate
        # If we want 10 items, we might need to fetch 100 or 500 candidates
        # to ensure we find matching brands.
        k_search = k * 50 
        
        found_ids = []
        
        # Search HNSW
        D, I = self.hnsw_index.search(query_vec.reshape(1, -1), k_search)
        
        for nid in I[0]:
            if nid == -1: continue
            
            node_data = self.graph.nodes[nid]
            
            # Attribute Check
            match_brand = node_data.get('brand') == target_attribute
            match_cat = node_data.get('category') == target_attribute
            
            if match_brand or match_cat:
                found_ids.append(nid)
                if len(found_ids) >= k: break
        
        return found_ids, time.perf_counter() - start

    def search_hnsw_bitmap(self, query_vec, target_attribute, k=10):
        """
        HNSW PRE-FILTER (Bitmap Approach):
        Passes a 'valid ID' selector into the HNSW search.
        """
        start = time.perf_counter()
        
        valid_ids = self.inverted_index.get(target_attribute, [])
        if not valid_ids:
            return [], time.perf_counter() - start
            
        # --- THE FIX IS HERE ---
        # 1. Force conversion to Int64 (standard for FAISS indices)
        # 2. Force 'ascontiguousarray' to ensure C-style memory layout
        valid_ids_arr = np.ascontiguousarray(valid_ids, dtype=np.int64)
        
        # 3. Construct IDSelectorArray safely
        selector = faiss.IDSelectorArray(len(valid_ids_arr), faiss.swig_ptr(valid_ids_arr))
        # -----------------------
        
        params = faiss.SearchParametersHNSW()
        params.sel = selector
        
        D, I = self.hnsw_index.search(query_vec.reshape(1, -1), k, params=params)
        
        found_ids = [nid for nid in I[0] if nid != -1]
        
        return found_ids, time.perf_counter() - start

    # =========================================
    # TYPE 2: STRUCTURE FILTERING (Neighbors)
    # =========================================

    def search_pre_filter_graph(self, query_vec, anchor_id, k=5):
        """Graph Traversal FIRST -> Vector Search"""
        start = time.perf_counter()
        neighbors = list(self.graph.neighbors(anchor_id))
        if not neighbors: return [], 0
        
        # SAFETY CHECK: Ensure neighbors are valid indices
        valid_indices = [i for i in neighbors if i < len(self.vectors)]
        if not valid_indices: return [], 0
            
        subset_vecs = self.vectors[valid_indices]
        temp_index = faiss.IndexFlatL2(self.d)
        temp_index.add(subset_vecs)
        D, I = temp_index.search(query_vec.reshape(1, -1), min(k, len(valid_indices)))
        found_ids = [valid_indices[i] for i in I[0] if i != -1]
        return found_ids, time.perf_counter() - start

    def search_post_filter(self, query_vec, target_attribute, k=10):
        """
        Iterative Post-Filter: 
        Automatically expands the search window if the first batch 
        doesn't contain enough matching items.
        """
        start = time.perf_counter()
        
        # Start with a reasonable guess (50x oversample)
        oversample_factor = 50 
        max_oversample = 500  # Safety cap
        
        found_ids = []
        seen_indices = set() 
        
        while len(found_ids) < k:
            k_search = k * oversample_factor
            
            D, I = self.vector_index.search(query_vec.reshape(1, -1), k_search)
            
            for nid in I[0]:
                if nid == -1: continue
                
                if nid in seen_indices:
                    continue
                seen_indices.add(nid)
                
                # --- UPDATE START: Unified Attribute Check ---
                node_data = self.graph.nodes[nid]
                
                # Check if target matches EITHER the brand OR the category
                # This allows the same function to work for "Sony" (Brand) and "Computers" (Category)
                match_brand = node_data.get('brand') == target_attribute
                match_cat = node_data.get('category') == target_attribute
                
                if match_brand or match_cat:
                    found_ids.append(nid)
                # --- UPDATE END ---
            
            if len(found_ids) >= k:
                found_ids = found_ids[:k]
                break
            
            if oversample_factor >= max_oversample:
                break
            
            oversample_factor *= 2
            
        return found_ids, time.perf_counter() - start