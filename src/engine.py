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

    # =============================================================================
    #                         SEARCH METHODS (5 Strategies)
    # =============================================================================
    #
    # METHOD 1: Bitmap HNSW (Pre-Filter)     - Baseline to beat, uses IDSelector
    # METHOD 2: Basic Post-Filter            - Naive baseline with fixed k_scan
    # METHOD 3: Adaptive Post-Filter         - Our innovation with iterative expansion
    # METHOD 4: Flat Brute Force             - Ground truth generator (100% accurate)
    # METHOD 5: Cached Partition (Future)    - Pre-built HNSW per filter (cache only)
    #
    # =============================================================================

    # =========================================================================
    # METHOD 1: BITMAP HNSW (Pre-Filter with IDSelector)
    # =========================================================================
    # This is the standard approach used by TigerVector and similar systems.
    # It passes a bitmap (IDSelector) to HNSW to skip invalid nodes during traversal.
    # This is the BASELINE TO BEAT.
    # =========================================================================
    
    def search_bitmap_hnsw(self, query_vec, filter_func: Callable, filter_desc: str, k=10):
        """
        METHOD 1: BITMAP HNSW (Pre-Filter with IDSelector)
        
        Standard approach used in TigerVector and similar vector databases.
        Creates a bitmap of valid IDs and passes it to HNSW search via IDSelector.
        The HNSW traversal skips nodes that don't pass the filter.
        
        This is the BASELINE that our adaptive method aims to beat.
        
        Args:
            query_vec: Query embedding vector
            filter_func: Lambda function to test if an item passes the filter
            filter_desc: Human-readable description of the filter
            k: Number of results to return
            
        Returns:
            (found_ids, time_in_seconds)
        """
        total_start = time.perf_counter()
        
        # === PHASE 1: BUILD BITMAP ===
        bitmap_start = time.perf_counter()
        
        # Scan all items to build bitmap of valid IDs
        valid_ids = [i for i in range(len(self.vectors)) if filter_func(i)]
        
        bitmap_time = (time.perf_counter() - bitmap_start) * 1000
        
        if not valid_ids:
            total_time = (time.perf_counter() - total_start) * 1000
            print(f"\n          [Bitmap HNSW] {filter_desc}")
            print(f"                  Bitmap build:      {bitmap_time:.3f}ms")
            print(f"                  Result:            0 items (empty set)")
            print(f"                  Total:             {total_time:.3f}ms")
            return [], total_time / 1000
        
        subset_size = len(valid_ids)
        selectivity = subset_size / len(self.vectors) * 100
        
        # === PHASE 2: SEARCH GLOBAL HNSW WITH BITMAP FILTER ===
        search_start = time.perf_counter()
        
        # Create bitmap selector using FAISS's built-in IDSelectorBatch
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
        
        print(f"\n          [Bitmap HNSW] {filter_desc}")
        print(f"                  Bitmap build:      {bitmap_time:.3f}ms (scanned {len(self.vectors):,} items)")
        print(f"                  Valid items:       {subset_size:,} ({selectivity:.1f}% selectivity)")
        print(f"                  HNSW search:       {search_time:.3f}ms (efSearch=200, with IDSelector)")
        print(f"                  Total:             {total_time:.3f}ms")
        print(f"                  Found:             {len(found_ids)}/{k} items")
        
        return found_ids, total_time / 1000

    # =========================================================================
    # METHOD 2: BASIC POST-FILTER (Naive Baseline)
    # =========================================================================
    # Simple approach: search HNSW with fixed k_scan, then filter results.
    # This is the naive baseline that breaks down on rare filters.
    # =========================================================================
    
    def search_post_filter_basic(self, query_vec, filter_func: Callable, filter_desc: str, k=10, k_scan=100):
        """
        METHOD 2: BASIC POST-FILTER (Naive Baseline)
        
        The simplest approach: search HNSW for k_scan candidates, then filter.
        This is fast for high selectivity but FAILS for rare filters.
        
        Args:
            query_vec: Query embedding vector
            filter_func: Lambda function to test if an item passes the filter
            filter_desc: Human-readable description of the filter
            k: Number of results to return
            k_scan: Number of candidates to fetch from HNSW (default 100)
            
        Returns:
            (found_ids, time_in_seconds)
        """
        total_start = time.perf_counter()
        
        # === PHASE 1: HNSW SEARCH ===
        search_start = time.perf_counter()
        D, I = self.hnsw_index.search(query_vec.reshape(1, -1), k_scan)
        search_time = (time.perf_counter() - search_start) * 1000
        
        # === PHASE 2: FILTER RESULTS ===
        filter_start = time.perf_counter()
        found_ids = []
        checked_count = 0
        
        for candidate_id in I[0]:
            if candidate_id == -1:
                continue
            checked_count += 1
            if filter_func(candidate_id):
                found_ids.append(int(candidate_id))
                if len(found_ids) >= k:
                    break
        
        filter_time = (time.perf_counter() - filter_start) * 1000
        total_time = (time.perf_counter() - total_start) * 1000
        
        # Calculate observed selectivity
        observed_selectivity = (len(found_ids) / checked_count * 100) if checked_count > 0 else 0
        
        print(f"\n          [Basic Post-Filter] {filter_desc}")
        print(f"                  k_scan:            {k_scan} (fixed)")
        print(f"                  HNSW search:       {search_time:.3f}ms")
        print(f"                  Filter loop:       {filter_time:.3f}ms ({checked_count} checked)")
        print(f"                  Observed density:  {observed_selectivity:.1f}%")
        print(f"                  Total:             {total_time:.3f}ms")
        print(f"                  Found:             {len(found_ids)}/{k} items")
        
        return found_ids, total_time / 1000

    # =========================================================================
    # METHOD 3: ADAPTIVE POST-FILTER (Our Innovation)
    # =========================================================================
    # Iterative expansion: start small, double k_scan if not enough results.
    # This guarantees high recall without always paying brute-force cost.
    # THIS IS OUR MAIN CONTRIBUTION - designed to beat Bitmap HNSW.
    # =========================================================================
    
    def search_post_filter_adaptive(self, query_vec, filter_func: Callable, filter_desc: str, k=10):
        """
        METHOD 3: ADAPTIVE POST-FILTER (Iterative Expansion)
        
        Our innovation: starts with small k_scan, doubles if not enough results.
        This guarantees high recall without always paying full scan cost.
        
        The Logic:
        1. Start with k_search = k * 5 (optimistic assumption)
        2. Search HNSW, filter results
        3. If found < k items, DOUBLE k_search and retry
        4. Repeat until k items found or entire index scanned
        
        This is designed to BEAT Bitmap HNSW for medium/high selectivity filters.
        
        Args:
            query_vec: Query embedding vector
            filter_func: Lambda function to test if an item passes the filter
            filter_desc: Human-readable description of the filter
            k: Number of results to return
            
        Returns:
            (found_ids, time_in_seconds)
        """
        total_start = time.perf_counter()
        
        # Initial guess: assume ~20% selectivity
        k_search = k * 5
        max_search = len(self.vectors)
        
        found_ids = []
        attempts = 0
        final_selectivity = 0.0
        total_checked = 0
        
        while len(found_ids) < k:
            attempts += 1
            
            # Cap k_search at dataset size
            if k_search > max_search:
                k_search = max_search
            
            # 1. HNSW Search
            D, I = self.hnsw_index.search(query_vec.reshape(1, -1), k_search)
            
            # 2. Filter Results
            # Reset found_ids because deeper search may find closer valid neighbors
            found_ids = []
            checked_count = 0
            
            for candidate_id in I[0]:
                if candidate_id == -1:
                    continue
                checked_count += 1
                if filter_func(candidate_id):
                    found_ids.append(int(candidate_id))
                    if len(found_ids) >= k:
                        break
            
            total_checked = checked_count
            
            # Exit conditions
            if len(found_ids) >= k:
                final_selectivity = (len(found_ids) / checked_count * 100) if checked_count > 0 else 0
                break
            
            if k_search >= max_search:
                # Scanned everything, stop
                final_selectivity = (len(found_ids) / checked_count * 100) if checked_count > 0 else 0
                break
            
            # EXPAND: Double the search scope
            k_search *= 2
        
        total_time = (time.perf_counter() - total_start) * 1000
        
        print(f"\n          [Adaptive Post-Filter] {filter_desc}")
        print(f"                  Attempts:          {attempts} (final k_search: {k_search})")
        print(f"                  Items checked:     {total_checked}")
        print(f"                  Observed density:  {final_selectivity:.2f}%")
        print(f"                  Total:             {total_time:.3f}ms")
        print(f"                  Found:             {len(found_ids)}/{k} items")
        
        return found_ids, total_time / 1000

    # =========================================================================
    # METHOD 4: FLAT BRUTE FORCE (Ground Truth Generator)
    # =========================================================================
    # Pre-filter all items, then compute exact distances.
    # This is 100% accurate but slow. Used only for ground truth generation.
    # May beat Bitmap HNSW for extremely rare filters (tiny valid set).
    # =========================================================================
    
    def search_flat_brute_force(self, query_vec, filter_func: Callable, filter_desc: str, k=10):
        """
        METHOD 4: FLAT BRUTE FORCE (Ground Truth Generator)
        
        The gold standard: filter all items, compute exact L2 distances.
        100% accurate but slow. Used for:
        - Generating ground truth for experiments
        - Extreme rare filters where valid set is tiny
        
        Args:
            query_vec: Query embedding vector
            filter_func: Lambda function to test if an item passes the filter
            filter_desc: Human-readable description of the filter
            k: Number of results to return
            
        Returns:
            (found_ids, time_in_seconds)
        """
        total_start = time.perf_counter()
        
        # === PHASE 1: FILTER ALL ITEMS ===
        filter_start = time.perf_counter()
        valid_ids = [i for i in range(len(self.vectors)) if filter_func(i)]
        filter_time = (time.perf_counter() - filter_start) * 1000
        
        if not valid_ids:
            total_time = (time.perf_counter() - total_start) * 1000
            print(f"\n          [Flat Brute Force] {filter_desc}")
            print(f"                  Filter scan:       {filter_time:.3f}ms (0 items found)")
            print(f"                  Total:             {total_time:.3f}ms")
            return [], total_time / 1000
        
        selectivity = len(valid_ids) / len(self.vectors) * 100
        
        # === PHASE 2: COMPUTE EXACT DISTANCES ===
        search_start = time.perf_counter()
        subset_vectors = self.vectors[valid_ids]
        
        # L2 squared distance
        diff = subset_vectors - query_vec.reshape(1, -1)
        dists = np.sum(diff**2, axis=1)
        
        # Get top-k
        k_actual = min(k, len(valid_ids))
        if len(dists) > k_actual:
            top_indices = np.argpartition(dists, k_actual)[:k_actual]
            top_indices = top_indices[np.argsort(dists[top_indices])]
        else:
            top_indices = np.argsort(dists)
        
        found_ids = [valid_ids[i] for i in top_indices]
        search_time = (time.perf_counter() - search_start) * 1000
        
        total_time = (time.perf_counter() - total_start) * 1000
        
        print(f"\n          [Flat Brute Force] {filter_desc}")
        print(f"                  Filter scan:       {filter_time:.3f}ms ({len(valid_ids):,} items, {selectivity:.1f}%)")
        print(f"                  Distance calc:     {search_time:.3f}ms ({len(valid_ids):,} vectors)")
        print(f"                  Total:             {total_time:.3f}ms")
        print(f"                  Found:             {len(found_ids)}/{k} items")
        
        return found_ids, total_time / 1000

    # =========================================================================
    # METHOD 5: CACHED PARTITION (Future Development)
    # =========================================================================
    # Uses pre-built HNSW indexes for specific filters (e.g., "price>100").
    # Only works if the exact filter is cached. Does NOT fall back.
    # This is the theoretical upper bound but impractical for arbitrary filters.
    # =========================================================================
    
    def search_cached_partition(self, query_vec, cache_key: str, k=10):
        """
        METHOD 5: CACHED PARTITION (Future Development)
        
        Uses pre-built HNSW indexes for specific filters.
        Only works if the EXACT filter is cached. Returns None on cache miss.
        
        This is the theoretical UPPER BOUND for speed but:
        - Requires pre-building HNSW for every possible filter
        - Storage explosion for many filters
        - Impractical for arbitrary numeric thresholds
        
        Future development: 
        - Pre-cache common filters at startup
        - Only use this method on cache hits
        - Fall back to other methods on cache miss
        
        Args:
            query_vec: Query embedding vector
            cache_key: The exact cache key (e.g., "price>100", "brand=Sony")
            k: Number of results to return
            
        Returns:
            (found_ids, time_in_seconds) if cache hit
            (None, 0.0) if cache miss
        """
        total_start = time.perf_counter()
        
        # Check if exact cache key exists
        if cache_key not in self.attribute_hnsw_cache:
            print(f"\n          [Cached Partition] Cache MISS for '{cache_key}'")
            return None, 0.0
        
        # Cache HIT - use pre-built index
        cached_index, cached_ids = self.attribute_hnsw_cache[cache_key]
        
        search_start = time.perf_counter()
        k_actual = min(k, len(cached_ids))
        D, I = cached_index.search(query_vec.reshape(1, -1), k_actual)
        
        # Map local indices back to global IDs
        found_ids = [cached_ids[i] for i in I[0] if i != -1 and i < len(cached_ids)]
        
        search_time = (time.perf_counter() - search_start) * 1000
        total_time = (time.perf_counter() - total_start) * 1000
        
        print(f"\n          [Cached Partition] Cache HIT for '{cache_key}'")
        print(f"                  Cached items:      {len(cached_ids):,}")
        print(f"                  HNSW search:       {search_time:.3f}ms")
        print(f"                  Total:             {total_time:.3f}ms")
        print(f"                  Found:             {len(found_ids)}/{k} items")
        
        return found_ids, total_time / 1000

    # =========================================================================
    # LEGACY ALIASES (For backward compatibility with existing experiments)
    # =========================================================================
    
    def search_pre_filter_flat_numeric(self, query_vec, filter_func: Callable, filter_desc: str, k=10):
        """Legacy alias for search_flat_brute_force"""
        return self.search_flat_brute_force(query_vec, filter_func, filter_desc, k)
    
    def search_bitmap_pre_filter(self, query_vec, filter_type: str, threshold, k=10):
        """
        Legacy alias that creates filter_func from filter_type and threshold.
        Maps to search_bitmap_hnsw.
        """
        if filter_type == 'price':
            filter_func = lambda i: self.item_prices[i] > threshold
            filter_desc = f"price > {threshold}"
        elif filter_type == 'rating':
            filter_func = lambda i: self.item_ratings[i] > threshold
            filter_desc = f"rating > {threshold}"
        elif filter_type == 'brand':
            filter_func = lambda i: self.item_brands[i] == threshold
            filter_desc = f"brand = {threshold}"
        elif filter_type == 'category':
            filter_func = lambda i: self.item_categories[i] == threshold
            filter_desc = f"category = {threshold}"
        else:
            return [], 0.0
        
        return self.search_bitmap_hnsw(query_vec, filter_func, filter_desc, k)