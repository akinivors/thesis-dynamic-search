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
        self.vector_index = faiss.IndexFlatL2(self.d)
        self.vectors = []
        
        # Optimization Maps
        self.brand_to_ids = {}      # Maps Brand Name -> List of Node IDs
        self.asin_map = {}          # Maps ASIN/Parent_ASIN -> Node ID
        self.brand_counts = {}      # Cached stats for the Adaptive Optimizer
        
        print("   [System] Loading embedding model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_data(self, filepath: str, limit: int = None):
        """
        Loads Data + Metadata + Edges.
        
        --- CRITICAL CHANGE LOG ---
        Updated to support the meta_Books.jsonl schema:
        - Item ID key changed from 'asin' to 'parent_asin'.
        - Edge key changed from ['also_buy', 'also_view'] to ['bought_together'].
        - Brand key changed from 'brand' to 'author' (using author as the main attribute).
        """
        
        # --- NEW SCHEMA KEYS ---
        ITEM_ID_KEY = 'parent_asin'
        TITLE_KEY = 'title'
        ATTRIBUTE_KEY = 'author'  # Using 'author' as the main filter attribute
        EDGE_KEY = 'bought_together'
        
        print(f"   [System] Reading dataset from {filepath}...")
        titles = []
        attributes = [] # Stores author names
        item_ids = []   # Stores parent_asin
        raw_edges = [] # Store (source_id, list_of_target_ids)
        
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # --- SCHEMA FIX FOR BOOKS (Focus on parent_asin and author) ---
                    pid = data.get(ITEM_ID_KEY)
                    title = data.get(TITLE_KEY)
                    # NOTE: We use the AUTHOR for the 'brand' filter attribute
                    author_data = data.get(ATTRIBUTE_KEY) 
                    
                    # Extract author name from the dictionary structure if it exists
                    author_name = author_data.get('name') if isinstance(author_data, dict) else None
                    
                    # We strictly require PID, Title, and Author
                    if pid and title and author_name:
                        item_ids.append(pid)
                        titles.append(title)
                        attributes.append(author_name)
                        
                        # --- EDGE COLLECTION (Using bought_together) ---
                        targets = data.get(EDGE_KEY)
                        
                        if targets and isinstance(targets, list):
                            # Targets are assumed to be parent_asin IDs
                            raw_edges.append((count, targets))
                            
                        count += 1
                        if limit and count >= limit: break
                        
                        if count % 50000 == 0:
                            print(f"            ...loaded {count} raw items...")
                            
                except: continue

        # --- DIAGNOSTIC PRINT 1 ---
        print(f"   [DEBUG] Captured {len(raw_edges)} potential connections from raw file.")

        # --- VECTORIZATION ---
        total_items = len(titles)
        if total_items == 0:
             print("   [ERROR] No data was loaded. Check JSONL format/keys.")
             self.vectors = np.array([])
             return # Exit to avoid ValueError
             
        print(f"   [System] Vectorizing {total_items} titles...")
        
        batch_size = 5000
        all_embeddings = []
        
        for start in range(0, total_items, batch_size):
            end = min(start + batch_size, total_items)
            batch_emb = self.model.encode(titles[start:end], show_progress_bar=False)
            all_embeddings.append(batch_emb)
            print(f"            ...encoded batch {end}/{total_items}...")
            
        self.vectors = np.vstack(all_embeddings).astype('float32')
        faiss.normalize_L2(self.vectors)
        self.vector_index.add(self.vectors)

        # --- GRAPH BUILDING ---
        print("   [System] Building Metadata Graph & Indices...")
        self.brand_counts = {}
        for i, pid in enumerate(item_ids):
            attribute_name = attributes[i]
            
            # Add to NetworkX
            self.graph.add_node(i, parent_asin=pid, attribute=attribute_name, title=titles[i])
            # Add to Lookup Maps
            self.asin_map[pid] = i
            
            if attribute_name not in self.brand_to_ids:
                self.brand_to_ids[attribute_name] = []
            self.brand_to_ids[attribute_name].append(i)
            self.brand_counts[attribute_name] = self.brand_counts.get(attribute_name, 0) + 1

        # --- EDGE LINKING ---
        print("   [System] Linking Graph Edges...")
        edge_count = 0
        missed_links = 0
        
        for source_id, target_pids in raw_edges:
            # target_pids are the parent_asin IDs of connected items
            for t_pid in target_pids:
                if t_pid in self.asin_map:
                    target_id = self.asin_map[t_pid]
                    # We continue to use the default edge for the 'bought together' links
                    self.graph.add_edge(source_id, target_id, edge_type="RELATED") 
                    edge_count += 1
                else:
                    missed_links += 1
        
        # --- DIAGNOSTIC PRINT 2 ---
        print(f"   [DEBUG] Linking Complete.")
        print(f"           Edges Created: {edge_count}")
        print(f"           Dangling Links (Target not in dataset): {missed_links}")
        
        print(f"   [System] Ready. Loaded {len(item_ids)} Nodes and {edge_count} Edges.")
        
    # --- The following methods must also be updated to use the 'attribute' key ---

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
        start = time.perf_counter()
        # Changed brand_to_ids lookup
        allow_list = self.brand_to_ids.get(target_attribute, []) 
        if not allow_list: return [], 0
        
        # ... rest of the pre-filter logic is the same ...
        
        # Note: The output tuple must be updated below if you want to track the method used.
        # This function returns (found_ids, elapsed_time)

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
        """The Thesis Contribution: Automatically picks Pre vs Post"""
        # Changed brand_counts lookup
        count = self.brand_counts.get(target_attribute, 0) 
        THRESHOLD = 100 # Tuned based on your experiment
        
        if count < THRESHOLD:
            return self.search_pre_filter(query_vec, target_attribute, k) + ("PRE-FILTER",)
        else:
            return self.search_post_filter(query_vec, target_attribute, k) + ("POST-FILTER",)

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

    def search_post_filter_graph(self, query_vec, anchor_id, k=5):
        """Vector Search FIRST -> Graph Validation"""
        start = time.perf_counter()
        valid_neighbors = set(self.graph.neighbors(anchor_id))
        if not valid_neighbors: return [], 0
        
        oversample = 200
        D, I = self.vector_index.search(query_vec.reshape(1, -1), k * oversample)
        found_ids = []
        for nid in I[0]:
            if nid == -1: continue
            if nid in valid_neighbors:
                found_ids.append(nid)
                if len(found_ids) >= k: break
        return found_ids, time.perf_counter() - start