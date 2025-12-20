import json
import time
import numpy as np
import networkx as nx
import faiss
from sentence_transformers import SentenceTransformer
from collections import Counter
from typing import List, Dict, Any, Callable





class ThesisEngine:
    def __init__(self):
        self.d = 384
        self.graph = nx.Graph()
        
        # 1. Exact Index (Baseline)
        self.vector_index = faiss.IndexFlatL2(self.d)
        
        # 2. HNSW Index (The Challenger)
        self.hnsw_index = faiss.IndexHNSWFlat(self.d, 64) 
        self.hnsw_index.hnsw.efConstruction = 80
        self.hnsw_index.hnsw.efSearch = 150
        
        self.vectors = None 
        
        # --- UNIFIED INDEX ---
        self.inverted_index = {}      
        self.value_counts = {}      
        self.asin_map = {}          
        
        # --- FAST ATTRIBUTE ARRAYS (For Post-Filter) ---
        self.item_brands = None       
        self.item_categories = None
        self.item_prices = None       
        self.item_ratings = None      
        
        # --- BITMAP INDEXES (For TigerVector-style Pre-Filter) ---
        self.price_bitmaps = {}
        self.rating_bitmaps = {}
        
        # Caching (for Standard Pre-Filter HNSW)
        self.attribute_hnsw_cache = {}
        
        # Cost Coefficients
        self.cost_per_filter_item = 0.000003 
        self.cost_index_batch = 0.020         
        
        print("   [System] Loading embedding model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_data(self, filepath: str, limit: int = None):
        """Loads Data with numeric fields"""
        ITEM_ID_KEY = 'asin'
        TITLE_KEY = 'title'
        KEY_BRAND = 'brand'
        KEY_CAT = 'main_cat'
        EDGE_KEY = 'also_buy'   
        KEY_PRICE = 'price'
        KEY_RATING = 'average_rating'
        
        print(f"   [System] Reading dataset from {filepath}...")
        titles = []
        item_ids = []   
        raw_edges = [] 
        
        meta_brands = []
        meta_cats = []
        prices = []
        ratings = []
        
        count = 0
        loaded_count = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    pid = data.get(ITEM_ID_KEY)
                    title = data.get(TITLE_KEY)
                    
                    if pid and title:
                        item_ids.append(pid)
                        titles.append(title)
                        
                        meta_brands.append(data.get(KEY_BRAND, "Unknown"))
                        meta_cats.append(data.get(KEY_CAT, "Unknown"))
                        
                        # Extract price
                        price_str = data.get(KEY_PRICE, "0")
                        if isinstance(price_str, str):
                            price_str = price_str.replace('$', '').replace(',', '')
                        try:
                            price = float(price_str)
                        except:  
                            price = 0.0
                        prices.append(price)
                        
                        # Extract rating
                        rating = float(data.get(KEY_RATING, 0.0))
                        ratings.append(rating)
                        
                        targets = data.get(EDGE_KEY, [])
                        if isinstance(targets, dict):
                            flat = []
                            for v in targets.values(): flat.extend(v)
                            targets = flat
                        if targets and isinstance(targets, list):
                            raw_edges.append((loaded_count, targets))
                        
                        loaded_count += 1
                        
                    count += 1
                    if limit and loaded_count >= limit:  break
                    if count % 50000 == 0:
                        print(f"            ...  processed {count} lines (loaded {loaded_count})...")
                            
                except json.JSONDecodeError: continue

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
        
        # Add to HNSW Index
        print("   [System] Building HNSW Graph Index (This may take a moment)...")
        self.hnsw_index.add(self.vectors)

        # --- STORE ATTRIBUTES AS ARRAYS ---
        print("   [System] Building attribute arrays for fast lookups...")
        self.item_brands = np.array(meta_brands, dtype=object)
        self.item_categories = np.array(meta_cats, dtype=object)
        self.item_prices = np.array(prices, dtype=np.float32)
        self.item_ratings = np.array(ratings, dtype=np.float32)
        
        # Print stats
        print(f"      Price range: ${np.min(self.item_prices):.2f} - ${np.max(self.item_prices):.2f}")
        print(f"      Rating range: {np.min(self.item_ratings):.2f} - {np.max(self.item_ratings):.2f}")

        # --- GRAPH & UNIFIED INDEX BUILDING ---
        print("   [System] Building Unified Index (Brands + Categories)...")
        self.graph.clear()
        self.inverted_index = {}
        self.value_counts = {}
        self.asin_map = {}
        
        for i, pid in enumerate(item_ids):
            b = meta_brands[i]
            c = meta_cats[i]
            
            self.graph.add_node(i, asin=pid, brand=b, category=c, title=titles[i], 
                              price=prices[i], rating=ratings[i])
            self.asin_map[pid] = i
            
            if b: 
                if b not in self.inverted_index:  self.inverted_index[b] = []
                self.inverted_index[b].append(i)
                self.value_counts[b] = self.value_counts.get(b, 0) + 1
            
            if c and c != b:
                if c not in self.inverted_index: self.inverted_index[c] = []
                self.inverted_index[c].append(i)
                self.value_counts[c] = self.value_counts.get(c, 0) + 1

        # --- BUILD BITMAP INDEXES ---
        self._build_bitmap_indexes()

        # --- EDGE LINKING ---
        print("   [System] Linking Graph Edges...")
        for source_id, target_pids in raw_edges:
            for t_pid in target_pids:
                if t_pid in self.asin_map:
                    target_id = self.asin_map[t_pid]
                    self.graph.add_edge(source_id, target_id, edge_type="RELATED") 
        
        print(f"   [System] Ready.  Indexed {len(self.inverted_index)} unique attributes.")

    def get_cache_stats(self):
        """Return detailed cache statistics"""
        categorical_keys = []
        numeric_keys = []
        
        for key in self.attribute_hnsw_cache.keys():
            if '>' in key:
                numeric_keys.append(key)
            else:
                categorical_keys.append(key)
        
        return {
            "categorical_cached": categorical_keys,
            "categorical_count": len(categorical_keys),
            "numeric_cached": numeric_keys,
            "numeric_count": len(numeric_keys),
            "total_cache_size": len(self.attribute_hnsw_cache),
        }
        
    def _build_bitmap_indexes(self):
        """
        Build bitmap indexes AND pre-built HNSW indexes for common numeric thresholds.
        This is the TRUE TigerVector approach:  build once at load time, reuse forever! 
        """
        print("   [System] Building bitmap indexes for numeric filters...")
        
        # Price thresholds
        price_thresholds = [25, 50, 75, 100, 150, 200, 300, 500, 1000]
        
        for threshold in price_thresholds: 
            bitmap_start = time.perf_counter()
            
            # Create bitmap (which items match?)
            bitmap_indices = np.where(self.item_prices > threshold)[0]
            bitmap = set(bitmap_indices)
            
            self.price_bitmaps[threshold] = bitmap
            
            subset_size = len(bitmap_indices)
            selectivity = subset_size / len(self.vectors) * 100
            
            # PRE-BUILD HNSW INDEX for this threshold (if enough items)
            if subset_size > 100:  # Only build if substantial
                print(f"      Price > ${threshold}: {subset_size:,} items ({selectivity:.1f}%) - Building HNSW...", end="", flush=True)
                
                build_start = time.perf_counter()
                subset_vectors = self.vectors[bitmap_indices]
                
                # Build dedicated HNSW index
                price_index = faiss.IndexHNSWFlat(self.d, 32)  # Smaller M for faster build
                price_index.hnsw.efConstruction = 100
                price_index.hnsw.efSearch = 150
                price_index.add(subset_vectors)
                
                build_time = (time.perf_counter() - build_start) * 1000
                
                # Cache the index with the bitmap
                cache_key = f"price>{threshold}"
                self.attribute_hnsw_cache[cache_key] = (price_index, list(bitmap_indices))
                
                print(f" Done ({int(build_time)}ms)")
            else:
                print(f"      Price > ${threshold}: {subset_size:,} items ({selectivity:.1f}%) - Too few, using flat search")
        
        # Rating thresholds
        rating_thresholds = [3.0, 3.5, 4.0, 4.5]
        
        for threshold in rating_thresholds:
            bitmap_start = time.perf_counter()
            
            # Create bitmap (which items match?)
            bitmap_indices = np.where(self.item_ratings > threshold)[0]
            bitmap = set(bitmap_indices)
            
            self.rating_bitmaps[threshold] = bitmap
            
            subset_size = len(bitmap_indices)
            selectivity = subset_size / len(self.vectors) * 100
            
            # PRE-BUILD HNSW INDEX for this threshold (if enough items)
            if subset_size > 100:  # Only build if substantial
                print(f"      Rating > {threshold}: {subset_size:,} items ({selectivity:.1f}%) - Building HNSW...", end="", flush=True)
                
                build_start = time.perf_counter()
                subset_vectors = self.vectors[bitmap_indices]
                
                # Build dedicated HNSW index
                rating_index = faiss.IndexHNSWFlat(self.d, 32)  # Smaller M for faster build
                rating_index.hnsw.efConstruction = 100
                rating_index.hnsw.efSearch = 150
                rating_index.add(subset_vectors)
                
                build_time = (time.perf_counter() - build_start) * 1000
                
                # Cache the index with the bitmap
                cache_key = f"rating>{threshold}"
                self.attribute_hnsw_cache[cache_key] = (rating_index, list(bitmap_indices))
                
                print(f" Done ({int(build_time)}ms)")
            else:
                print(f"      Rating > {threshold}: {subset_size:,} items ({selectivity:.1f}%) - Too few, using flat search")

    def get_details(self, node_ids: List[int]) -> List[str]:
        """Helper to fetch human-readable details"""
        details = []
        for nid in node_ids:
            data = self.graph.nodes[nid]
            details.append(f"[{data.get('brand', 'N/A')}] ${data.get('price', 0):.2f} ★{data.get('rating', 0):.1f} {data['title'][:60]}...") 
        return details

    # =========================================
    # STRATEGY 1: STANDARD PRE-FILTER HNSW (Categorical)
    # =========================================
    
    def search_standard_pre_filter_hnsw(self, query_vec, target_attribute, k=10):
        """
        STANDARD Pre-Filter HNSW (for categorical attributes like brand/category)
        Builds and caches HNSW index per attribute.
        """
        total_start = time.perf_counter()
        build_time = 0.0
        
        cache_lookup_start = time.perf_counter()
        valid_ids = self.inverted_index.get(target_attribute, [])
        
        if not valid_ids:
            return [], time.perf_counter() - total_start, 0.0
        
        subset_size = len(valid_ids)
        
        # Check if cached
        if target_attribute not in self.attribute_hnsw_cache:
            print(f"         [Building HNSW index for '{target_attribute}' ({subset_size} items)...]", end="", flush=True)
            build_start = time.perf_counter()
            
            subset_vectors = self.vectors[valid_ids]
            subset_index = faiss.IndexHNSWFlat(self.d, 64)
            subset_index.hnsw.efConstruction = 200
            subset_index.hnsw.efSearch = 200
            subset_index.add(subset_vectors)
            
            self.attribute_hnsw_cache[target_attribute] = (subset_index, valid_ids)
            
            build_time = time.perf_counter() - build_start
            print(f" Done ({int(build_time*1000)}ms)")
        
        subset_index, cached_ids = self.attribute_hnsw_cache[target_attribute]
        cache_lookup_time = (time.perf_counter() - cache_lookup_start) * 1000
        
        search_start = time.perf_counter()
        k_actual = min(k, subset_size)
        D, I = subset_index.search(query_vec.reshape(1, -1), k_actual)
        hnsw_search_time = (time.perf_counter() - search_start) * 1000
        
        mapping_start = time.perf_counter()
        found_ids = [cached_ids[i] for i in I[0] if i != -1 and i < len(cached_ids)]
        mapping_time = (time.perf_counter() - mapping_start) * 1000
        
        total_time = (time.perf_counter() - total_start) * 1000
        
        print(f"\n          [Standard Pre-Filter HNSW] {target_attribute} ({subset_size} items)")
        print(f"                  Cache lookup:      {cache_lookup_time:.3f}ms")
        print(f"                  HNSW search:      {hnsw_search_time:.3f}ms")
        print(f"                  Index mapping:    {mapping_time:.3f}ms")
        print(f"                  Total:            {total_time:.3f}ms")
        
        return found_ids, total_time / 1000, build_time

    # =========================================
    # STRATEGY 2: BITMAP PRE-FILTER HNSW (TigerVector)
    # =========================================
    
    def search_bitmap_pre_filter_hnsw(self, query_vec, filter_type, threshold, k=10):
        """
        BITMAP Pre-Filter HNSW (TRUE TigerVector approach)
        Uses pre-built bitmap AND pre-built HNSW indexes. 
        """
        total_start = time.perf_counter()
        
        # === PHASE 1: BITMAP LOOKUP ===
        bitmap_start = time.perf_counter()
        
        # Find closest pre-built threshold
        if filter_type == 'price':
            available = sorted(self.price_bitmaps.keys())
            closest = min(available, key=lambda x:  abs(x - threshold))
            bitmap = self.price_bitmaps[closest]
            cache_key = f"price>{closest}"
        elif filter_type == 'rating': 
            available = sorted(self.rating_bitmaps.keys())
            closest = min(available, key=lambda x: abs(x - threshold))
            bitmap = self.rating_bitmaps[closest]
            cache_key = f"rating>{closest}"
        else:
            return [], time.perf_counter() - total_start, 0.0
        
        # If requested threshold differs from cached, apply additional filter
        if filter_type == 'price': 
            if closest < threshold:
                valid_ids = [i for i in bitmap if self.item_prices[i] > threshold]
            else:
                valid_ids = list(bitmap)
        else:  # rating
            if closest < threshold:
                valid_ids = [i for i in bitmap if self.item_ratings[i] > threshold]
            else:
                valid_ids = list(bitmap)
        
        bitmap_time = (time.perf_counter() - bitmap_start) * 1000
        
        if not valid_ids:
            print(f"\n          [Bitmap Pre-Filter HNSW] {filter_type.title()} > {threshold}")
            print(f"                  Bitmap lookup:      {bitmap_time:.3f}ms")
            print(f"                  Result:             0 items (empty set)")
            print(f"                  Total:             {bitmap_time:.3f}ms")
            return [], bitmap_time / 1000, 0.0
        
        subset_size = len(valid_ids)
        selectivity = subset_size / len(self.vectors)
        
        # === PHASE 2: CHECK IF WE HAVE CACHED HNSW INDEX ===
        search_start = time.perf_counter()
        
        if cache_key in self.attribute_hnsw_cache:
            # USE PRE-BUILT CACHED INDEX ✅
            cached_index, cached_ids = self.attribute_hnsw_cache[cache_key]
            
            # If exact match (no additional filtering), use cached IDs directly
            if closest == threshold: 
                # Search cached HNSW
                k_actual = min(k, len(cached_ids))
                D, I = cached_index.search(query_vec.reshape(1, -1), k_actual)
                found_ids = [cached_ids[i] for i in I[0] if i != -1 and i < len(cached_ids)]
                search_time = (time.perf_counter() - search_start) * 1000
                
                total_time = (time.perf_counter() - total_start) * 1000
                
                print(f"\n          [Bitmap Pre-Filter HNSW] {filter_type.title()} > {threshold}")
                print(f"                  Bitmap lookup:     {bitmap_time:.3f}ms")
                print(f"                  Cache hit:         YES ✅ (using pre-built HNSW)")
                print(f"                  HNSW search:       {search_time:.3f}ms (on {len(cached_ids):,} cached items)")
                print(f"                  Total:             {total_time:.3f}ms")
                print(f"                  Found:              {len(found_ids)}/{k} items")
                
                return found_ids, total_time / 1000, 0.0
            
            else:
                # Need to filter cached results further
                # Map valid_ids to cached index positions
                cached_set = set(cached_ids)
                valid_in_cache = [i for i, cid in enumerate(cached_ids) if cid in set(valid_ids)]
                
                if len(valid_in_cache) < 1000:
                    # Few items - use flat search on valid subset
                    subset_vectors = self.vectors[valid_ids]
                    diff = subset_vectors - query_vec.reshape(1, -1)
                    dists = np.sum(diff**2, axis=1)
                    
                    k_actual = min(k, len(valid_ids))
                    if len(dists) > k_actual:
                        top_indices = np.argpartition(dists, k_actual)[:k_actual]
                        top_indices = top_indices[np.argsort(dists[top_indices])]
                    else:
                        top_indices = np.argsort(dists)
                    
                    found_ids = [valid_ids[i] for i in top_indices]
                    search_time = (time.perf_counter() - search_start) * 1000
                    
                    total_time = (time.perf_counter() - total_start) * 1000
                    
                    print(f"\n          [Bitmap Pre-Filter HNSW] {filter_type.title()} > {threshold}")
                    print(f"                  Bitmap lookup:     {bitmap_time:.3f}ms (threshold mismatch:  {closest} vs {threshold})")
                    print(f"                  Additional filter:  {subset_size:,} items")
                    print(f"                  Search method:     Flat (subset <1000)")
                    print(f"                  Search time:       {search_time:.3f}ms")
                    print(f"                  Total:             {total_time:.3f}ms")
                    print(f"                  Found:             {len(found_ids)}/{k} items")
                    
                    return found_ids, total_time / 1000, 0.0
        
        # === PHASE 3: NO CACHED INDEX - FALLBACK ===
        # This should rarely happen if we pre-built for all common thresholds
        
        if subset_size < 1000:
            # Small subset - use flat search
            subset_vectors = self.vectors[valid_ids]
            diff = subset_vectors - query_vec.reshape(1, -1)
            dists = np.sum(diff**2, axis=1)
            
            k_actual = min(k, len(valid_ids))
            if len(dists) > k_actual:
                top_indices = np.argpartition(dists, k_actual)[:k_actual]
                top_indices = top_indices[np.argsort(dists[top_indices])]
            else:
                top_indices = np.argsort(dists)
            
            found_ids = [valid_ids[i] for i in top_indices]
            search_time = (time.perf_counter() - search_start) * 1000
            search_method = "Flat"
            build_time = 0.0
            
        else:
            # Large subset - must build temp HNSW (FALLBACK - should be rare!)
            print(f"      ⚠️  Building temp HNSW for {subset_size:,} items (no cached index).. .", end="", flush=True)
            
            build_start = time.perf_counter()
            subset_vectors = self.vectors[valid_ids]
            temp_index = faiss.IndexHNSWFlat(self.d, 32)
            temp_index.hnsw.efConstruction = 40
            temp_index.hnsw.efSearch = 100
            temp_index.add(subset_vectors)
            build_time = (time.perf_counter() - build_start) * 1000
            
            print(f" Done ({int(build_time)}ms)")
            
            k_actual = min(k, subset_size)
            D, I = temp_index.search(query_vec.reshape(1, -1), k_actual)
            found_ids = [valid_ids[i] for i in I[0] if i != -1 and i < len(valid_ids)]
            search_time = (time.perf_counter() - search_start - build_time/1000) * 1000
            search_method = "Temp HNSW (fallback)"
        
        total_time = (time.perf_counter() - total_start) * 1000
        
        print(f"\n          [Bitmap Pre-Filter HNSW] {filter_type.title()} > {threshold}")
        print(f"                  Bitmap lookup:     {bitmap_time:.3f}ms")
        print(f"                  Cache hit:         NO ⚠️ (using {search_method})")
        print(f"                  Subset size:       {subset_size:,} items ({selectivity*100:.1f}%)")
        if build_time > 0:
            print(f"                  Temp build:         {build_time:.3f}ms")
        print(f"                  Search time:       {search_time:.3f}ms")
        print(f"                  Total:             {total_time:.3f}ms")
        print(f"                  Found:             {len(found_ids)}/{k} items")
        
        return found_ids, total_time / 1000, build_time / 1000
    # =========================================
    # STRATEGY 3: POST-FILTER HNSW (ChromaDB)
    # =========================================
    
    def search_post_filter_hnsw_numeric(self, query_vec, filter_func, filter_name, k=10, adaptive_k_search=True):
        """
        Post-filter HNSW search with ADAPTIVE k_search for better recall.
        
        Args:
            adaptive_k_search: If True, use smart k_search calculation (NEW FIXED VERSION)
                            If False, use old fixed k_search (OLD BROKEN VERSION)
        """
        total_start = time.perf_counter()
        
        # === PHASE 1: SELECTIVITY ESTIMATION ===
        setup_start = time.perf_counter()
        sample_size = 1000
        sample_indices = np.random.choice(len(self.vectors), min(sample_size, len(self.vectors)), replace=False)
        matching_sample = sum(1 for i in sample_indices if filter_func(i))
        est_selectivity_pct = (matching_sample / len(sample_indices)) * 100
        setup_time = (time.perf_counter() - setup_start) * 1000
        
        # === PHASE 2: ADAPTIVE K_SEARCH ===
        if adaptive_k_search:
            # NEW FIXED VERSION:  Adaptive k_search based on selectivity
            if est_selectivity_pct < 0.1:
                # Ultra-low selectivity: search 50K candidates to find 10 results
                k_search = 50000
            elif est_selectivity_pct < 0.5:
                # Very low:  search 20K candidates
                k_search = 20000
            elif est_selectivity_pct < 1.0:
                # Low: search 10K candidates  
                k_search = 10000
            elif est_selectivity_pct < 2.0:
                # Medium-low: search 5K candidates
                k_search = 5000
            elif est_selectivity_pct < 5.0:
                # Medium: search 2K candidates
                k_search = 2000
            elif est_selectivity_pct < 10.0:
                # Medium-high: search 1K candidates
                k_search = 1000
            else:
                # High: use formula
                k_search = max(300, int(k * 100 / est_selectivity_pct))
        else:
            # OLD BROKEN VERSION: Fixed k_search
            if est_selectivity_pct < 1.0:
                k_search = 1000
            elif est_selectivity_pct < 5.0:
                k_search = 500
            else:
                k_search = max(150, int(k * 100 / est_selectivity_pct))
        
        k_search = min(k_search, len(self.vectors))
        
        # === PHASE 3: HNSW SEARCH ===
        search_start = time.perf_counter()
        D, I = self.hnsw_index.search(query_vec. reshape(1, -1), k_search)
        search_time = (time.perf_counter() - search_start) * 1000
        
        # === PHASE 4: FILTER RESULTS ===
        filter_start = time.perf_counter()
        found_ids = []
        checked_count = 0
        
        for candidate_id in I[0]:
            if candidate_id == -1 or candidate_id >= len(self.vectors):
                continue
            checked_count += 1
            if filter_func(candidate_id):
                found_ids.append(int(candidate_id))
                if len(found_ids) >= k:
                    break
        
        filter_time = (time. perf_counter() - filter_start) * 1000
        total_time = (time.perf_counter() - total_start) * 1000
        
        actual_selectivity = (len([i for i in I[0] if i != -1 and i < len(self.vectors) and filter_func(i)]) / checked_count * 100) if checked_count > 0 else 0
        avg_filter_time = filter_time / checked_count if checked_count > 0 else 0
        
        version = "FIXED" if adaptive_k_search else "OLD"
        
        print(f"\n          [Post-Filter HNSW - {version}] {filter_name}")
        print(f"                  Est. selectivity:  {est_selectivity_pct:.1f}% (sampled {sample_size} items)")
        print(f"                  Actual selectivity: {actual_selectivity:.1f}%")
        print(f"                  Setup:              {setup_time:.3f}ms")
        print(f"                  k_search:          {k_search}")
        print(f"                  HNSW search:       {search_time:.3f}ms (efSearch=128)")
        print(f"                  Filter loop:       {filter_time:.3f}ms ({checked_count} items checked)")
        print(f"                  Avg filter/item:   {avg_filter_time:.6f}ms")
        print(f"                  Total:             {total_time:.3f}ms")
        print(f"                  Found:             {len(found_ids)}/{k} items")
        
        return found_ids, total_time / 1000

    # =========================================
    # STRATEGY 4: PRE-FILTER FLAT (Baseline)
    # =========================================


    def search_bitmap_filter_global_hnsw(self, query_vec, filter_type, threshold, k=10):
        """
        TRUE TIGERVECTOR BITMAP PRE-FILTER
        Searches the GLOBAL HNSW index while filtering with bitmap during traversal. 
        Uses FAISS's built-in IDSelector functionality.
        """
        total_start = time.perf_counter()
        
        # === PHASE 1: CREATE BITMAP ===
        bitmap_start = time.perf_counter()
        
        # Find closest pre-computed threshold
        if filter_type == 'price':
            available = sorted(self.price_bitmaps.keys())
            if not available:
                print(f"\n          [TRUE Bitmap Filter] No price bitmaps available")
                return [], 0.0
            
            closest = min(available, key=lambda x:  abs(x - threshold))
            base_bitmap = self.price_bitmaps[closest]
        elif filter_type == 'rating': 
            available = sorted(self.rating_bitmaps.keys())
            if not available:
                print(f"\n          [TRUE Bitmap Filter] No rating bitmaps available")
                return [], 0.0
            
            closest = min(available, key=lambda x: abs(x - threshold))
            base_bitmap = self.rating_bitmaps[closest]
        else:
            return [], 0.0
        
        # Apply exact threshold if different from closest
        if filter_type == 'price':
            if closest != threshold:
                valid_ids = [i for i in base_bitmap if self.item_prices[i] > threshold]
            else:
                valid_ids = list(base_bitmap)
        else:  # rating
            if closest != threshold:
                valid_ids = [i for i in base_bitmap if self.item_ratings[i] > threshold]
            else:
                valid_ids = list(base_bitmap)
        
        bitmap_time = (time.perf_counter() - bitmap_start) * 1000
        
        if not valid_ids:
            total_time = (time.perf_counter() - total_start) * 1000
            print(f"\n          [TRUE Bitmap Filter - Global HNSW] {filter_type.title()} > {threshold}")
            print(f"                  Bitmap lookup:      {bitmap_time:.3f}ms")
            print(f"                  Result:             0 items (empty set)")
            print(f"                  Total:              {total_time:.3f}ms")
            return [], total_time / 1000
        
        subset_size = len(valid_ids)
        selectivity = subset_size / len(self.vectors) * 100
        
        # === PHASE 2: SEARCH GLOBAL HNSW WITH BITMAP FILTER ===
        search_start = time.perf_counter()
        
        # Create bitmap selector using FAISS's built-in IDSelectorBatch
        # This requires a sorted array of valid IDs
        valid_ids_array = np.array(sorted(valid_ids), dtype=np.int64)
        selector = faiss.IDSelectorBatch(valid_ids_array)
        
        # Create search parameters with selector
        params = faiss.SearchParametersHNSW()
        params.sel = selector
        params.efSearch = 200  # Higher efSearch for filtered search
        
        # Search global HNSW index with bitmap filtering
        k_actual = min(k, subset_size)
        D, I = self.hnsw_index.search(query_vec.reshape(1, -1), k_actual, params=params)
        
        # Extract results
        found_ids = [int(i) for i in I[0] if i != -1 and i < len(self.vectors)]
        
        search_time = (time.perf_counter() - search_start) * 1000
        total_time = (time.perf_counter() - total_start) * 1000
        
        # Print detailed stats
        print(f"\n          [TRUE Bitmap Filter - Global HNSW] {filter_type.title()} > {threshold}")
        print(f"                  Bitmap lookup:     {bitmap_time:.3f}ms (threshold: {threshold}, closest cached: {closest})")
        print(f"                  Valid items:       {subset_size:,} ({selectivity:.1f}% selectivity)")
        print(f"                  Index used:        GLOBAL HNSW ({len(self.vectors):,} items) with bitmap filter ✅")
        print(f"                  Search time:       {search_time:.3f}ms (efSearch=200)")
        print(f"                  Total:              {total_time:.3f}ms")
        print(f"                  Found:              {len(found_ids)}/{k} items")
        
        return found_ids, total_time / 1000



    
    def search_pre_filter_flat_numeric(self, query_vec, filter_func: Callable, filter_desc: str, k=10):
        """
        PRE-FILTER FLAT (Baseline - brute force)
        Filter all items then compute distances.
        """
        total_start = time.perf_counter()
        
        # Filter all items
        filter_start = time.perf_counter()
        valid_ids = [i for i in range(len(self.vectors)) if filter_func(i)]
        filter_time = (time.perf_counter() - filter_start) * 1000
        
        if not valid_ids:
            return [], time.perf_counter() - total_start
        
        # Brute force search
        search_start = time.perf_counter()
        subset_vectors = self.vectors[valid_ids]
        diff = subset_vectors - query_vec.reshape(1, -1)
        dists = np.sum(diff**2, axis=1)
        
        k_actual = min(k, len(valid_ids))
        if len(dists) > k_actual:
            top_indices = np.argpartition(dists, k_actual)[:k_actual]
            top_indices = top_indices[np.argsort(dists[top_indices])]
        else:
            top_indices = np.argsort(dists)
        
        found_ids = [valid_ids[i] for i in top_indices]
        search_time = (time.perf_counter() - search_start) * 1000
        
        total_time = (time.perf_counter() - total_start) * 1000
        selectivity = len(valid_ids) / len(self.vectors)
        
        print(f"\n          [Pre-Filter Flat] {filter_desc}")
        print(f"                  Filter scan:       {filter_time:.3f}ms (found {len(valid_ids):,} items, {selectivity*100:.1f}%)")
        print(f"                  Brute force:        {search_time:.3f}ms ({len(valid_ids):,} distances)")
        print(f"                  Total:             {total_time:.3f}ms")
        print(f"                  Found:             {len(found_ids)}/{k} items")
        
        return found_ids, total_time / 1000