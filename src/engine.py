import json
import time
import numpy as np
import networkx as nx
import faiss
from sentence_transformers import SentenceTransformer
from collections import Counter

class ThesisEngine:
    def __init__(self):
        self.d = 384
        self.graph = nx.Graph()
        self.vector_index = faiss.IndexFlatL2(self.d)
        self.vectors = []
        
        # Optimization Maps
        self.brand_to_ids = {}      # Maps Brand Name -> List of Node IDs
        self.asin_map = {}          # Maps ASIN -> Node ID
        self.brand_counts = {}      # Cached stats for the Adaptive Optimizer
        
        print("   [System] Loading embedding model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_data(self, filepath, limit=None):
        """
        Loads Data + Metadata + Edges (Amazon V2 Schema).
        Includes Debug prints and Progress Bars.
        """
        print(f"   [System] Reading dataset from {filepath}...")
        titles = []
        brands = []
        asins = []
        raw_edges = [] # Store (source_id, list_of_target_asins)
        
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # We strictly require Title, Brand, and ASIN
                    if 'title' in data and 'brand' in data and 'asin' in data:
                        titles.append(data['title'])
                        brands.append(data['brand'])
                        asins.append(data['asin'])
                        
                        # --- SCHEMA FIX FOR AMAZON V2 ---
                        # Look for 'also_buy' and 'also_view' at the root level
                        targets = []
                        targets.extend(data.get('also_buy', []))
                        targets.extend(data.get('also_view', []))
                        
                        if targets:
                            raw_edges.append((count, targets))
                            
                        count += 1
                        if limit and count >= limit: break
                        
                        # Log reading progress
                        if count % 50000 == 0:
                            print(f"            ...loaded {count} raw items...")
                except: continue

        # --- DIAGNOSTIC PRINT 1 ---
        print(f"   [DEBUG] Captured {len(raw_edges)} potential connections from raw file.")

        # --- VECTORIZATION (With Progress Log) ---
        total_items = len(titles)
        print(f"   [System] Vectorizing {total_items} titles...")
        
        batch_size = 5000
        all_embeddings = []
        
        for start in range(0, total_items, batch_size):
            end = min(start + batch_size, total_items)
            # Encode batch
            batch_emb = self.model.encode(titles[start:end], show_progress_bar=False)
            all_embeddings.append(batch_emb)
            
            # CRITICAL: Print progress so we know it's not frozen
            print(f"            ...encoded batch {end}/{total_items}...")
            
        self.vectors = np.vstack(all_embeddings).astype('float32')
        faiss.normalize_L2(self.vectors)
        self.vector_index.add(self.vectors)

        # --- GRAPH BUILDING ---
        print("   [System] Building Metadata Graph & Indices...")
        for i, asin in enumerate(asins):
            b = brands[i]
            # Add to NetworkX
            self.graph.add_node(i, asin=asin, brand=b, title=titles[i])
            # Add to Lookup Maps
            self.asin_map[asin] = i
            if b not in self.brand_to_ids:
                self.brand_to_ids[b] = []
            self.brand_to_ids[b].append(i)
            self.brand_counts[b] = self.brand_counts.get(b, 0) + 1

        # --- EDGE LINKING ---
        print("   [System] Linking Graph Edges...")
        edge_count = 0
        missed_links = 0
        for source_id, target_asins in raw_edges:
            for t_asin in target_asins:
                # Only link if the target actually exists in our loaded dataset
                if t_asin in self.asin_map:
                    target_id = self.asin_map[t_asin]
                    self.graph.add_edge(source_id, target_id)
                    edge_count += 1
                else:
                    missed_links += 1
        
        # --- DIAGNOSTIC PRINT 2 ---
        print(f"   [DEBUG] Linking Complete.")
        print(f"           Edges Created: {edge_count}")
        print(f"           Dangling Links (Target not in dataset): {missed_links}")
        
        print(f"   [System] Ready. Loaded {len(asins)} Nodes and {edge_count} Edges.")

    def get_details(self, node_ids):
        """Helper to fetch human-readable details for a list of IDs"""
        details = []
        for nid in node_ids:
            data = self.graph.nodes[nid]
            details.append(f"[{data['brand']}] {data['title'][:80]}...")
        return details

    def get_contextual_brands(self, query_vec, search_k=5000):
        """Analyzes vector neighborhood to find appropriate test brands (Common/Mid/Rare)"""
        D, I = self.vector_index.search(query_vec.reshape(1, -1), search_k)
        found_brands = []
        for nid in I[0]:
            if nid == -1: continue
            b = self.graph.nodes[nid].get('brand')
            if b: found_brands.append(b)
        c = Counter(found_brands)
        
        def pick(min_c, max_c):
            opts = [b for b, cnt in c.items() if min_c <= cnt <= max_c]
            return (opts[0], c[opts[0]]) if opts else (None, 0)

        return {
            "Common": pick(200, 5000),
            "Mid-Tier": pick(20, 50),
            "Rare": pick(2, 5)
        }

    # =========================================
    # TYPE 1: ATTRIBUTE FILTERING (Brand)
    # =========================================
    
    def search_pre_filter(self, query_vec, target_brand, k=10):
        start = time.perf_counter()
        allow_list = self.brand_to_ids.get(target_brand, [])
        if not allow_list: return [], 0
        
        # SAFETY CHECK: Ensure we don't access out-of-bounds vectors
        valid_indices = [i for i in allow_list if i < len(self.vectors)]
        if not valid_indices: return [], 0
        
        subset_vecs = self.vectors[valid_indices]
        temp_index = faiss.IndexFlatL2(self.d)
        temp_index.add(subset_vecs)
        
        D, I = temp_index.search(query_vec.reshape(1, -1), min(k, len(valid_indices)))
        
        found_ids = [valid_indices[i] for i in I[0] if i != -1]
        return found_ids, time.perf_counter() - start

    def search_post_filter(self, query_vec, target_brand, k=10):
        start = time.perf_counter()
        oversample = 50
        D, I = self.vector_index.search(query_vec.reshape(1, -1), k * oversample)
        found_ids = []
        for nid in I[0]:
            if nid == -1: continue
            if self.graph.nodes[nid].get('brand') == target_brand:
                found_ids.append(nid)
                if len(found_ids) >= k: break
        return found_ids, time.perf_counter() - start

    def search_adaptive(self, query_vec, target_brand, k=10):
        """The Thesis Contribution: Automatically picks Pre vs Post"""
        count = self.brand_counts.get(target_brand, 0)
        THRESHOLD = 100 # Tuned based on your experiment
        
        if count < THRESHOLD:
            return self.search_pre_filter(query_vec, target_brand, k) + ("PRE-FILTER",)
        else:
            return self.search_post_filter(query_vec, target_brand, k) + ("POST-FILTER",)

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