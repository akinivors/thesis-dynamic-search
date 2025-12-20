import numpy as np
from src.engine import ThesisEngine
import time

def main():
    print("="*120)
    print("TRUE TIGERVECTOR BITMAP FILTERING TEST")
    print("="*120)
    print("\nComparing:")
    print("  1. Pre-Filter with SEPARATE cached indexes (our previous 'bitmap')")
    print("  2. TRUE Bitmap Filter with GLOBAL HNSW + IDSelector (TigerVector style)")
    print("  3. Post-Filter HNSW (baseline)")
    print("="*120)
    
    # Initialize
    print("\nInitializing...")
    engine = ThesisEngine()
    engine.load_data('data/meta_Electronics.json')  # Use subset for speed
    
    # Test query
    query_text = "high quality laptop computer"
    query_vec = engine.model.encode([query_text])[0].astype('float32')
    
    test_cases = [
        {"name": "price > $100 (cached)", "threshold": 100, "field": "price"},
        {"name": "price > $137 (not cached)", "threshold": 137, "field": "price"},
        {"name": "price > $200 (cached)", "threshold": 200, "field": "price"},
        {"name": "price > $333. 33 (not cached)", "threshold": 333.33, "field": "price"},
    ]
    
    print(f"\nQuery: '{query_text}'")
    print("="*120)
    
    results = []
    
    for test_num, test in enumerate(test_cases, 1):
        print(f"\n{'='*120}")
        print(f"TEST {test_num}/{len(test_cases)}: {test['name']}")
        print(f"{'='*120}")
        
        threshold = test['threshold']
        filter_func = lambda nid: engine.item_prices[nid] > threshold
        
        # Count matching items
        matching = sum(1 for i in range(len(engine.vectors)) if filter_func(i))
        selectivity = matching / len(engine. vectors) * 100
        print(f"  Matching items: {matching:,} ({selectivity:.2f}%)")
        
        # === STRATEGY 1: PRE-FILTER WITH SEPARATE INDEXES ===
        print(f"\n  >>> STRATEGY 1: Pre-Filter with Separate Cached Indexes")
        pre_ids, pre_time, _ = engine.search_bitmap_pre_filter_hnsw(
            query_vec, test['field'], threshold, k=10
        )
        pre_time_ms = pre_time * 1000
        print(f"      Results: {len(pre_ids)}/10")
        
        # === STRATEGY 2: TRUE BITMAP FILTER (GLOBAL HNSW) ===
        print(f"\n  >>> STRATEGY 2: TRUE Bitmap Filter (Global HNSW + IDSelector)")
        true_ids, true_time = engine.search_bitmap_filter_global_hnsw(
            query_vec, test['field'], threshold, k=10
        )
        true_time_ms = true_time * 1000
        print(f"      Results:  {len(true_ids)}/10")
        
        # === STRATEGY 3: POST-FILTER ===
        print(f"\n  >>> STRATEGY 3: Post-Filter HNSW")
        post_ids, post_time = engine.search_post_filter_hnsw_numeric(
            query_vec, filter_func, f"{test['name']}", k=10
        )
        post_time_ms = post_time * 1000
        print(f"      Results: {len(post_ids)}/10")
        
        # === COMPARISON ===
        print(f"\n  {'─'*110}")
        print(f"  COMPARISON:")
        print(f"    Pre-Filter (separate):   {pre_time_ms: 7.2f}ms")
        print(f"    TRUE Bitmap (global):    {true_time_ms:7.2f}ms")
        print(f"    Post-Filter:              {post_time_ms:7.2f}ms")
        
        times = {
            "Pre-Filter (separate)": pre_time_ms,
            "TRUE Bitmap (global)": true_time_ms,
            "Post-Filter":  post_time_ms
        }
        winner = min(times, key=times.get)
        print(f"    Winner: {winner}")
        print(f"  {'─'*110}")
        
        results.append({
            "test": test['name'],
            "matching": matching,
            "selectivity": selectivity,
            "pre_filter_ms": pre_time_ms,
            "true_bitmap_ms": true_time_ms,
            "post_filter_ms": post_time_ms,
            "winner": winner,
        })
    
    # === SUMMARY ===
    print("\n" + "="*120)
    print("📊 SUMMARY")
    print("="*120)
    print(f"\n{'Test':<30} | {'Pre-Sep (ms)':<12} | {'TRUE Bitmap':<12} | {'Post (ms)':<10} | Winner")
    print("-"*100)
    for r in results:
        print(f"{r['test']:<30} | {r['pre_filter_ms']:>11.2f} | {r['true_bitmap_ms']:>11.2f} | {r['post_filter_ms']:>9.2f} | {r['winner']}")
    
    print("\n" + "="*120)
    print("🎯 KEY INSIGHTS")
    print("="*120)
    print("\n1. Pre-Filter with Separate Indexes:")
    print(f"   - Cached:  {np.mean([r['pre_filter_ms'] for r in results[: 2]]):.2f}ms avg")
    print(f"   - Builds dedicated HNSW per threshold")
    print(f"   - Fast for cached, slow for uncached")
    
    print("\n2. TRUE Bitmap Filter (Global HNSW):")
    print(f"   - All queries: {np.mean([r['true_bitmap_ms'] for r in results]):.2f}ms avg")
    print(f"   - Uses ONE global index for all filters")
    print(f"   - Memory efficient (no separate indexes)")
    print(f"   - Consistent performance")
    
    print("\n3. Post-Filter:")
    print(f"   - All queries: {np.mean([r['post_filter_ms'] for r in results]):.2f}ms avg")
    print(f"   - Baseline approach")
    print("="*120)

if __name__ == "__main__":
    main()