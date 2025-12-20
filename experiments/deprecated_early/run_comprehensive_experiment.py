import numpy as np
from src.engine import ThesisEngine
import time
import json
from datetime import datetime

def main():
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"experiment_results_{timestamp}.log"
    results_file = f"experiment_results_{timestamp}.json"
    
    def log(message):
        """Log to both console and file"""
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')
    
    log("="*120)
    log("COMPREHENSIVE EXPERIMENT: Pre-Filter vs Post-Filter")
    log("="*120)
    log(f"\nTimestamp: {datetime.now()}")
    log(f"Log file: {log_file}")
    log(f"Results file: {results_file}")
    
    log("\nTesting:")
    log("  Strategies: Pre-Filter HNSW vs Post-Filter HNSW")
    log("  Filter Types:  Categorical vs Numeric")
    log("  Cache Status: Cached vs Not Cached")
    log("="*120)
    
    # Initialize
    log("\n[1/4] Initializing engine...")
    engine = ThesisEngine()
    engine.load_data('data/meta_Electronics.json')
    
    # Get cache stats
    log("\n[2/4] Examining what's cached...")
    cache_stats = engine.get_cache_stats()
    log(f"\nCache Statistics:")
    log(f"  Total cached indexes: {cache_stats['total_cache_size']}")
    log(f"  Categorical cached: {cache_stats['categorical_count']}")
    log(f"  Numeric cached: {cache_stats['numeric_count']}")
    
    log(f"\n  Categorical cache keys (first 10):")
    for key in list(cache_stats['categorical_cached'])[:10]:
        if '>' not in key:  # Only show categorical
            log(f"    - {key}")
    
    log(f"\n  Numeric cache keys:")
    for key in cache_stats['numeric_cached']: 
        log(f"    - {key}")
    
    # Test query
    query_text = "high quality laptop computer"
    query_vec = engine.model.encode([query_text])[0].astype('float32')
    
    log(f"\n[3/4] Test query: '{query_text}'")
    
    # Define test cases
    # Define test cases
    # Define test cases
    test_cases = [
        # === CATEGORICAL TESTS ===
        {
            "name": "Categorical - Cached (Electronics)",
            "type": "categorical",
            "cached": True,
            "attribute":  "Electronics",
            "filter_func": lambda nid: "Electronics" in engine.inverted_index and nid in engine.inverted_index["Electronics"],
        },
        {
            "name": "Categorical - Cached (Computers)",
            "type": "categorical",
            "cached":  True,
            "attribute": "Computers",
            "filter_func": lambda nid: "Computers" in engine.inverted_index and nid in engine.inverted_index["Computers"],
        },
        {
            "name": "Categorical - NOT Cached (Rare)",
            "type": "categorical",
            "cached": False,
            "attribute": "RareTestAttribute999",
            "filter_func": lambda nid: "RareTestAttribute999" in engine.inverted_index and nid in engine.inverted_index["RareTestAttribute999"],
        },
        
        # === NUMERIC TESTS ===
        {
            "name": "Numeric - Cached (price > $100)",
            "type": "numeric",
            "cached": True,
            "threshold": 100,
            "field": "price",
            "filter_func": lambda nid: engine.item_prices[nid] > 100,
        },
        {
            "name": "Numeric - Cached (price > $200)",
            "type": "numeric",
            "cached": True,
            "threshold": 200,
            "field": "price",
            "filter_func": lambda nid:  engine.item_prices[nid] > 200,
        },
        {
            "name":  "Numeric - NOT Cached (price > $137)",
            "type": "numeric",
            "cached": False,
            "threshold": 137,
            "field": "price",
            "filter_func": lambda nid: engine.item_prices[nid] > 137,
        },
        {
            "name": "Numeric - NOT Cached (price > $333.33)",
            "type": "numeric",
            "cached": False,
            "threshold": 333.33,
            "field": "price",
            "filter_func": lambda nid: engine.item_prices[nid] > 333.33,
        },
    ]
    
    log(f"\n[4/4] Running {len(test_cases)} test cases...")
    log("="*120)
    
    all_results = []
    
    for test_num, test in enumerate(test_cases, 1):
        log(f"\n{'='*120}")
        log(f"TEST {test_num}/{len(test_cases)}:  {test['name']}")
        log(f"{'='*120}")
        log(f"  Type: {test['type']}")
        log(f"  Cached: {test['cached']}")
        
        # Count matching items
        matching_count = sum(1 for i in range(len(engine.vectors)) if test['filter_func'](i))
        selectivity = matching_count / len(engine.vectors) * 100
        
        log(f"  Matching items: {matching_count:,} ({selectivity:.2f}%)")
        
        # Check if actually in cache
        if test['type'] == 'categorical':
            cache_key = test['attribute']
            actually_cached = cache_key in engine.attribute_hnsw_cache
        else:
            cache_key = f"{test['field']}>{test['threshold']}"
            actually_cached = cache_key in engine.attribute_hnsw_cache
        
        log(f"  Cache key: '{cache_key}'")
        log(f"  Actually in cache: {actually_cached}")
        
        # === STRATEGY 1: PRE-FILTER HNSW ===
        log(f"\n  >>> STRATEGY 1: PRE-FILTER HNSW")
        
        pre_start = time.perf_counter()
        
        if test['type'] == 'categorical':
            # Use categorical pre-filter
            if actually_cached:
                pre_ids, pre_time = engine.search_standard_pre_filter_hnsw(
                    query_vec, test['attribute'], k=10
                )
            else:
                # Fallback:  build temp index
                log(f"      ⚠️  NOT IN CACHE - Building temp index...")
                filter_start = time.perf_counter()
                valid_ids = [i for i in range(len(engine.vectors)) if test['filter_func'](i)]
                filter_time = (time.perf_counter() - filter_start) * 1000
                
                if len(valid_ids) == 0:
                    pre_time = (time.perf_counter() - pre_start)
                    pre_ids = []
                    log(f"      No matching items - skipping index build")
                elif len(valid_ids) < 1000:
                    # Use flat search
                    subset_vectors = engine.vectors[valid_ids]
                    diff = subset_vectors - query_vec.reshape(1, -1)
                    dists = np.sum(diff**2, axis=1)
                    k_actual = min(10, len(valid_ids))
                    top_indices = np.argsort(dists)[:k_actual]
                    pre_ids = [valid_ids[i] for i in top_indices]
                    pre_time = (time.perf_counter() - pre_start)
                    log(f"      Used flat search (subset < 1000)")
                else:
                    # Build temp HNSW
                    import faiss
                    build_start = time.perf_counter()
                    subset_vectors = engine.vectors[valid_ids]
                    temp_index = faiss.IndexHNSWFlat(engine.d, 32)
                    temp_index.hnsw.efConstruction = 40
                    temp_index.hnsw.efSearch = 100
                    temp_index.add(subset_vectors)
                    build_time = (time.perf_counter() - build_start) * 1000
                    
                    D, I = temp_index.search(query_vec.reshape(1, -1), 10)
                    pre_ids = [valid_ids[i] for i in I[0] if i != -1]
                    pre_time = (time.perf_counter() - pre_start)
                    log(f"      Built temp HNSW:  {build_time:.1f}ms")
        
        else:  # numeric
            pre_ids, pre_time, build_time = engine.search_bitmap_pre_filter_hnsw(
                query_vec, test['field'], test['threshold'], k=10
            )
        
        pre_time_ms = pre_time * 1000 if isinstance(pre_time, float) else pre_time
        
        log(f"      Total time: {pre_time_ms:.2f}ms")
        log(f"      Results: {len(pre_ids)}/10 items")
        
        # === STRATEGY 2: POST-FILTER HNSW ===
        log(f"\n  >>> STRATEGY 2: POST-FILTER HNSW")
        
        filter_desc = f"{test['name']}"
        post_ids, post_time = engine.search_post_filter_hnsw_numeric(
            query_vec, test['filter_func'], filter_desc, k=10
        )
        
        post_time_ms = post_time * 1000
        
        log(f"      Total time: {post_time_ms:.2f}ms")
        log(f"      Results:  {len(post_ids)}/10 items")
        
        # === COMPARISON ===
        log(f"\n  {'─'*110}")
        log(f"  COMPARISON:")
        log(f"  {'─'*110}")
        log(f"    Pre-Filter:    {pre_time_ms: 8.2f}ms")
        log(f"    Post-Filter:  {post_time_ms:8.2f}ms")
        
        if pre_time_ms < post_time_ms:
            winner = "Pre-Filter"
            speedup = post_time_ms / pre_time_ms
        else:
            winner = "Post-Filter"
            speedup = pre_time_ms / post_time_ms
        
        log(f"    Winner: {winner} ({speedup:.1f}x faster)")
        log(f"  {'─'*110}")
        
        # Store results
        result = {
            "test_number": test_num,
            "name": test['name'],
            "type": test['type'],
            "cached_expected": test['cached'],
            "cached_actual": actually_cached,
            "cache_key": cache_key,
            "matching_items": matching_count,
            "selectivity_pct": selectivity,
            "pre_filter_ms": pre_time_ms,
            "post_filter_ms":  post_time_ms,
            "winner": winner,
            "speedup": speedup,
            "pre_filter_results": len(pre_ids),
            "post_filter_results": len(post_ids),
        }
        all_results.append(result)
    
    # === FINAL SUMMARY ===
    log("\n" + "="*120)
    log("📊 FINAL SUMMARY")
    log("="*120)
    
    log(f"\n{'Test':<40} | {'Type':<12} | {'Cached':<8} | {'Pre (ms)':<10} | {'Post (ms)':<10} | {'Winner':<15} | {'Speedup': <8}")
    log("-"*120)
    for r in all_results:
        log(f"{r['name']:<40} | {r['type']:<12} | {str(r['cached_actual']):<8} | {r['pre_filter_ms']: >9.2f} | {r['post_filter_ms']:>9.2f} | {r['winner']:<15} | {r['speedup']: >6.1f}x")
    
    # Group analysis
    log("\n" + "="*120)
    log("📈 GROUPED ANALYSIS")
    log("="*120)
    
    # Categorical cached
    cat_cached = [r for r in all_results if r['type'] == 'categorical' and r['cached_actual']]
    if cat_cached: 
        log(f"\n1.  CATEGORICAL - CACHED ({len(cat_cached)} tests):")
        log(f"   Pre-Filter avg:   {np.mean([r['pre_filter_ms'] for r in cat_cached]):.2f}ms")
        log(f"   Post-Filter avg: {np.mean([r['post_filter_ms'] for r in cat_cached]):.2f}ms")
        log(f"   Winner: {max(set([r['winner'] for r in cat_cached]), key=[r['winner'] for r in cat_cached].count)}")
    
    # Categorical not cached
    cat_not = [r for r in all_results if r['type'] == 'categorical' and not r['cached_actual']]
    if cat_not:
        log(f"\n2. CATEGORICAL - NOT CACHED ({len(cat_not)} tests):")
        log(f"   Pre-Filter avg:  {np.mean([r['pre_filter_ms'] for r in cat_not]):.2f}ms")
        log(f"   Post-Filter avg: {np.mean([r['post_filter_ms'] for r in cat_not]):.2f}ms")
        log(f"   Winner: {max(set([r['winner'] for r in cat_not]), key=[r['winner'] for r in cat_not].count)}")
    
    # Numeric cached
    num_cached = [r for r in all_results if r['type'] == 'numeric' and r['cached_actual']]
    if num_cached: 
        log(f"\n3. NUMERIC - CACHED ({len(num_cached)} tests):")
        log(f"   Pre-Filter avg:  {np.mean([r['pre_filter_ms'] for r in num_cached]):.2f}ms")
        log(f"   Post-Filter avg: {np.mean([r['post_filter_ms'] for r in num_cached]):.2f}ms")
        log(f"   Winner: {max(set([r['winner'] for r in num_cached]), key=[r['winner'] for r in num_cached].count)}")
    
    # Numeric not cached
    num_not = [r for r in all_results if r['type'] == 'numeric' and not r['cached_actual']]
    if num_not: 
        log(f"\n4. NUMERIC - NOT CACHED ({len(num_not)} tests):")
        log(f"   Pre-Filter avg:  {np.mean([r['pre_filter_ms'] for r in num_not]):.2f}ms")
        log(f"   Post-Filter avg: {np.mean([r['post_filter_ms'] for r in num_not]):.2f}ms")
        log(f"   Winner: {max(set([r['winner'] for r in num_not]), key=[r['winner'] for r in num_not].count)}")
    
    log("\n" + "="*120)
    log("🎯 KEY CONCLUSIONS")
    log("="*120)
    
    log("\n1. When Pre-Filter cache HIT:")
    cached_results = [r for r in all_results if r['cached_actual']]
    if cached_results: 
        pre_wins = sum(1 for r in cached_results if r['winner'] == 'Pre-Filter')
        log(f"   Pre-Filter wins: {pre_wins}/{len(cached_results)} tests")
        log(f"   Avg Pre-Filter:   {np.mean([r['pre_filter_ms'] for r in cached_results]):.2f}ms")
        log(f"   Avg Post-Filter: {np.mean([r['post_filter_ms'] for r in cached_results]):.2f}ms")
    
    log("\n2. When Pre-Filter cache MISS:")
    non_cached_results = [r for r in all_results if not r['cached_actual']]
    if non_cached_results: 
        pre_wins = sum(1 for r in non_cached_results if r['winner'] == 'Pre-Filter')
        log(f"   Pre-Filter wins: {pre_wins}/{len(non_cached_results)} tests")
        log(f"   Avg Pre-Filter:  {np.mean([r['pre_filter_ms'] for r in non_cached_results]):.2f}ms")
        log(f"   Avg Post-Filter: {np.mean([r['post_filter_ms'] for r in non_cached_results]):.2f}ms")
    
    log("\n" + "="*120)
    log(f"Experiment complete! Results saved to:")
    log(f"  - Log:      {log_file}")
    log(f"  - JSON:    {results_file}")
    log("="*120)
    
    # Save JSON
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "query": query_text,
            "cache_stats": cache_stats,
            "results": all_results,
        }, f, indent=2)

if __name__ == "__main__":
    main()
