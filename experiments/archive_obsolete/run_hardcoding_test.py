import numpy as np
from src.engine import ThesisEngine
import time

def main():
    print("="*120)
    print("EXPERIMENT 1: THE HARDCODING PROBLEM")
    print("="*120)
    print("\nQuestion:  What happens when users query thresholds NOT in our cache?")
    print("\nHardcoded cache thresholds:  $25, $50, $75, $100, $150, $200, $300, $500, $1000")
    print("="*120)
    
    # Initialize
    engine = ThesisEngine()
    engine.load_data('data/meta_Electronics.json')
    
    # Test query
    query_text = "high quality laptop computer"
    query_vec = engine.model.encode([query_text])[0].astype('float32')
    
    # Test cases:  Mix of cached and non-cached thresholds
    test_cases = [
        # CACHED (exact match)
        {"threshold": 100, "in_cache": True, "name": "Price > $100 (CACHED)"},
        {"threshold": 200, "in_cache": True, "name": "Price > $200 (CACHED)"},
        {"threshold": 500, "in_cache": True, "name": "Price > $500 (CACHED)"},
        
        # NOT CACHED (near cached values)
        {"threshold": 99, "in_cache": False, "name": "Price > $99 (NOT CACHED - near $100)"},
        {"threshold":  137, "in_cache": False, "name": "Price > $137 (NOT CACHED - between $100-$150)"},
        {"threshold": 225, "in_cache": False, "name": "Price > $225 (NOT CACHED - between $200-$300)"},
        {"threshold": 501, "in_cache": False, "name": "Price > $501 (NOT CACHED - near $500)"},
        
        # NOT CACHED (arbitrary values)
        {"threshold": 42.50, "in_cache": False, "name": "Price > $42.50 (NOT CACHED - arbitrary)"},
        {"threshold": 333.33, "in_cache":  False, "name": "Price > $333.33 (NOT CACHED - arbitrary)"},
        {"threshold": 777, "in_cache": False, "name": "Price > $777 (NOT CACHED - arbitrary)"},
    ]
    
    results = []
    
    print(f"\n{'='*120}")
    print(f"TESTING {len(test_cases)} PRICE THRESHOLDS")
    print(f"Query: '{query_text}'")
    print("="*120)
    
    for test_num, test in enumerate(test_cases, 1):
        threshold = test['threshold']
        
        print(f"\n{'='*120}")
        print(f"Test {test_num}/{len(test_cases)}: {test['name']}")
        print(f"{'='*120}")
        
        # Create filter function
        filter_func = lambda nid: engine.item_prices[nid] > threshold
        
        # Count how many items match
        matching_count = sum(1 for i in range(len(engine.vectors)) if filter_func(i))
        selectivity = matching_count / len(engine.vectors) * 100
        
        print(f"\n   Dataset stats:")
        print(f"      Threshold:     ${threshold}")
        print(f"      Matching items: {matching_count: ,} ({selectivity:.2f}%)")
        print(f"      In cache:     {'YES ✅' if test['in_cache'] else 'NO ❌'}")
        
        # === STRATEGY 1: BITMAP PRE-FILTER HNSW ===
        print(f"\n   >>> BITMAP PRE-FILTER HNSW:")
        bitmap_ids, bitmap_time, build_time = engine.search_bitmap_pre_filter_hnsw(
            query_vec, 'price', threshold, k=10
        )
        
        # === STRATEGY 2: POST-FILTER HNSW ===
        print(f"\n   >>> POST-FILTER HNSW:")
        post_ids, post_time = engine.search_post_filter_hnsw_numeric(
            query_vec, filter_func, f"Price > ${threshold}", k=10
        )
        
        # === COMPARISON ===
        print(f"\n{'='*120}")
        print(f"   RESULTS:")
        print(f"      Bitmap Pre-Filter:   {bitmap_time*1000:7.2f}ms")
        print(f"      Post-Filter HNSW:  {post_time*1000:7.2f}ms")
        
        if bitmap_time < post_time:
            winner = "Bitmap Pre-Filter"
            speedup = post_time / bitmap_time
        else:
            winner = "Post-Filter HNSW"
            speedup = bitmap_time / post_time
        
        print(f"      Winner: {winner} ({speedup:.1f}x faster)")
        print(f"{'='*120}")
        
        results.append({
            "threshold": threshold,
            "in_cache": test['in_cache'],
            "matching_items": matching_count,
            "selectivity": selectivity,
            "bitmap_time": bitmap_time * 1000,
            "post_time": post_time * 1000,
            "winner": winner,
            "speedup": speedup,
        })
    
    # === ANALYSIS ===
    print("\n" + "="*120)
    print("📊 SUMMARY:  Cached vs Non-Cached Performance")
    print("="*120)
    
    cached_results = [r for r in results if r['in_cache']]
    non_cached_results = [r for r in results if not r['in_cache']]
    
    print(f"\n{'='*120}")
    print("CACHED THRESHOLDS (Exact Match):")
    print(f"{'='*120}")
    print(f"| {'Threshold':<15} | Matching Items | Bitmap (ms) | Post (ms) | Winner | Speedup |")
    print(f"|{'-'*17}|{'-'*16}|{'-'*13}|{'-'*11}|{'-'*20}|{'-'*9}|")
    for r in cached_results:
        print(f"| ${r['threshold']:<14. 2f} | {r['matching_items']:>8,} ({r['selectivity']: >4.1f}%) | {r['bitmap_time']:>10.2f} | {r['post_time']: >8.2f} | {r['winner']:<18} | {r['speedup']: >6.1f}x |")
    
    avg_cached_bitmap = np.mean([r['bitmap_time'] for r in cached_results])
    avg_cached_post = np.mean([r['post_time'] for r in cached_results])
    
    print(f"\nAverage (Cached):")
    print(f"   Bitmap Pre-Filter: {avg_cached_bitmap:7.2f}ms")
    print(f"   Post-Filter HNSW:   {avg_cached_post:7.2f}ms")
    print(f"   Speedup:  Bitmap is {avg_cached_post/avg_cached_bitmap:.1f}x faster ✅")
    
    print(f"\n{'='*120}")
    print("NON-CACHED THRESHOLDS (NOT in hardcoded list):")
    print(f"{'='*120}")
    print(f"| {'Threshold':<15} | Matching Items | Bitmap (ms) | Post (ms) | Winner | Speedup |")
    print(f"|{'-'*17}|{'-'*16}|{'-'*13}|{'-'*11}|{'-'*20}|{'-'*9}|")
    for r in non_cached_results:
        print(f"| ${r['threshold']:<14.2f} | {r['matching_items']: >8,} ({r['selectivity']:>4.1f}%) | {r['bitmap_time']:>10.2f} | {r['post_time']:>8.2f} | {r['winner']:<18} | {r['speedup']:>6.1f}x |")
    
    avg_non_cached_bitmap = np.mean([r['bitmap_time'] for r in non_cached_results])
    avg_non_cached_post = np.mean([r['post_time'] for r in non_cached_results])
    
    print(f"\nAverage (Non-Cached):")
    print(f"   Bitmap Pre-Filter: {avg_non_cached_bitmap:7.2f}ms")
    print(f"   Post-Filter HNSW:   {avg_non_cached_post:7.2f}ms")
    
    if avg_non_cached_bitmap < avg_non_cached_post:
        print(f"   Speedup: Bitmap is {avg_non_cached_post/avg_non_cached_bitmap:.1f}x faster")
    else:
        print(f"   Speedup: Post is {avg_non_cached_bitmap/avg_non_cached_post:.1f}x faster ⚠️")
    
    print("\n" + "="*120)
    print("🔍 KEY QUESTIONS:")
    print("="*120)
    print("\n1. How does Bitmap Pre-Filter handle non-cached thresholds?")
    print("   → Does it use the nearest cached threshold?")
    print("   → Does it fall back to temp index building?")
    print("   → Does it fall back to Post-Filter?")
    print("\n2. Is there a performance degradation for non-cached thresholds?")
    print(f"   → Cached average:      {avg_cached_bitmap:. 2f}ms")
    print(f"   → Non-cached average: {avg_non_cached_bitmap:.2f}ms")
    
    if avg_non_cached_bitmap > avg_cached_bitmap * 2:
        print(f"   → ⚠️  SIGNIFICANT DEGRADATION ({avg_non_cached_bitmap/avg_cached_bitmap:.1f}x slower)")
    elif avg_non_cached_bitmap > avg_cached_bitmap * 1.2:
        print(f"   → ⚠️  Moderate degradation ({avg_non_cached_bitmap/avg_cached_bitmap:.1f}x slower)")
    else:
        print(f"   → ✅ Minimal degradation ({avg_non_cached_bitmap/avg_cached_bitmap:.1f}x slower)")
    
    print("\n3. Should we hardcode thresholds or use adaptive caching? ")
    bitmap_wins = sum(1 for r in non_cached_results if r['winner'] == 'Bitmap Pre-Filter')
    post_wins = sum(1 for r in non_cached_results if r['winner'] == 'Post-Filter HNSW')
    
    print(f"   → Non-cached queries:  Bitmap wins {bitmap_wins}/{len(non_cached_results)}, Post wins {post_wins}/{len(non_cached_results)}")
    
    if bitmap_wins > post_wins:
        print(f"   → ✅ Bitmap Pre-Filter still works well for non-cached thresholds")
        print(f"   → Recommendation:  Hardcoding is sufficient")
    else:
        print(f"   → ⚠️  Post-Filter performs better for non-cached thresholds")
        print(f"   → Recommendation: Need adaptive caching or use Post-Filter for arbitrary thresholds")
    
    print("="*120)

if __name__ == "__main__":
    main()
