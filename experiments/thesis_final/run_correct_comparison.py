import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import json
from datetime import datetime
from src.engine import ThesisEngine

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def run_correct_comparison():
    log("=" * 80)
    log("CORRECT THREE-WAY COMPARISON EXPERIMENT")
    log("Testing:  TRUE Bitmap vs Post-Filter OLD vs Post-Filter FIXED")
    log("=" * 80)
    
    # Load engine
    log("\n[1/6] Loading engine...")
    engine = ThesisEngine()
    
    log("\n[2/6] Loading data...")
    engine.load_data('data/meta_Electronics.json')  # Load all data
    
    log(f"\n  ✓ Loaded {len(engine.vectors)} items")
    log(f"  ✓ HNSW index built with {engine.hnsw_index.ntotal} vectors")
    
    # Test queries
    queries = [
        "laptop computer high quality",
        "wireless bluetooth headphones",
        "gaming mouse RGB",
        "USB cable charger",
        "portable external hard drive"
    ]
    
    # Price thresholds covering different selectivities
    # These should match some of your pre-built cached thresholds
    price_thresholds = [25, 50, 75, 100, 150, 200, 300, 500, 1000]
    
    log(f"\n[3/6] Testing {len(queries)} queries × {len(price_thresholds)} thresholds = {len(queries)*len(price_thresholds)} combinations")
    
    # Pre-compute selectivities
    log("\n[4/6] Computing selectivities...")
    selectivities = {}
    for threshold in price_thresholds: 
        count = np.sum(engine.item_prices > threshold)
        selectivity = (count / len(engine.item_prices)) * 100
        selectivities[threshold] = {
            'count': int(count),
            'percentage': round(selectivity, 2)
        }
        
        # Check if this threshold is cached
        cache_key = f"price>{threshold}"
        is_cached = cache_key in engine.attribute_hnsw_cache
        cached_mark = "✅ CACHED" if is_cached else "  "
        
        log(f"  price > ${threshold:4d}: {count:6,d} items ({selectivity:5.2f}%) {cached_mark}")
    
    # Show cache stats
    cache_stats = engine.get_cache_stats()
    log(f"\n  Cache Status:")
    log(f"    Numeric cached: {cache_stats['numeric_count']}")
    log(f"    Keys:  {cache_stats['numeric_cached']}")
    
    # Run experiments
    log("\n[5/6] Running experiments...")
    log("=" * 80)
    
    results = []
    total_tests = len(queries) * len(price_thresholds)
    test_num = 0
    
    for query_text in queries:
        log(f"\n{'='*80}")
        log(f"QUERY: '{query_text}'")
        log(f"{'='*80}")
        
        query_embedding = engine.model.encode([query_text])[0].astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
        
        for threshold in price_thresholds: 
            test_num += 1
            
            log(f"\n{'─'*80}")
            log(f"[TEST {test_num}/{total_tests}] Price > ${threshold} (Selectivity: {selectivities[threshold]['percentage']:.2f}%, {selectivities[threshold]['count']:,} items)")
            log(f"{'─'*80}")
            
            test_result = {
                'test_num': test_num,
                'query': query_text,
                'threshold': threshold,
                'selectivity_pct': selectivities[threshold]['percentage'],
                'matching_items': selectivities[threshold]['count']
            }
            
            # === METHOD 1: TRUE BITMAP (Global HNSW + IDSelectorBatch) ===
            log(f"\n>>> METHOD 1: TRUE Bitmap Filter (Global HNSW)")
            try:
                bitmap_ids, bitmap_time = engine.search_bitmap_filter_global_hnsw(
                    query_embedding, 'price', threshold, k=10
                )
                
                test_result['bitmap_time_ms'] = round(bitmap_time * 1000, 3)
                test_result['bitmap_results_found'] = len([i for i in bitmap_ids if i != -1])
                test_result['bitmap_ids'] = [int(i) for i in bitmap_ids if i != -1]
                
            except Exception as e:
                log(f"    ❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
                test_result['bitmap_time_ms'] = None
                test_result['bitmap_results_found'] = 0
                test_result['bitmap_error'] = str(e)
            
            # === METHOD 2: POST-FILTER OLD (adaptive_k_search=False) ===
            log(f"\n>>> METHOD 2: Post-Filter OLD (Fixed k_search)")
            try:
                # Create filter function
                def price_filter(idx):
                    return engine.item_prices[idx] > threshold
                
                post_old_ids, post_old_time = engine.search_post_filter_hnsw_numeric(
                    query_embedding, price_filter, f"price > ${threshold}", 
                    k=10, adaptive_k_search=False
                )
                
                test_result['post_old_time_ms'] = round(post_old_time * 1000, 3)
                test_result['post_old_results_found'] = len([i for i in post_old_ids if i != -1])
                test_result['post_old_ids'] = [int(i) for i in post_old_ids if i != -1]
                
            except Exception as e:
                log(f"    ❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
                test_result['post_old_time_ms'] = None
                test_result['post_old_results_found'] = 0
                test_result['post_old_error'] = str(e)
            
            # === METHOD 3: POST-FILTER FIXED (adaptive_k_search=True) ===
            log(f"\n>>> METHOD 3: Post-Filter FIXED (Adaptive k_search)")
            try:
                post_fixed_ids, post_fixed_time = engine.search_post_filter_hnsw_numeric(
                    query_embedding, price_filter, f"price > ${threshold}", 
                    k=10, adaptive_k_search=True
                )
                
                test_result['post_fixed_time_ms'] = round(post_fixed_time * 1000, 3)
                test_result['post_fixed_results_found'] = len([i for i in post_fixed_ids if i != -1])
                test_result['post_fixed_ids'] = [int(i) for i in post_fixed_ids if i != -1]
                
            except Exception as e:
                log(f"    ❌ ERROR: {e}")
                import traceback
                traceback. print_exc()
                test_result['post_fixed_time_ms'] = None
                test_result['post_fixed_results_found'] = 0
                test_result['post_fixed_error'] = str(e)
            
            # Quick summary
            log(f"\n{'─'*80}")
            log(f"RESULTS SUMMARY:")
            log(f"  TRUE Bitmap:    {test_result.get('bitmap_time_ms', 'ERROR'):>8} ms  |  Found: {test_result.get('bitmap_results_found', 0)}/10")
            log(f"  Post-OLD:      {test_result.get('post_old_time_ms', 'ERROR'):>8} ms  |  Found: {test_result.get('post_old_results_found', 0)}/10")
            log(f"  Post-FIXED:    {test_result.get('post_fixed_time_ms', 'ERROR'):>8} ms  |  Found: {test_result.get('post_fixed_results_found', 0)}/10")
            log(f"{'─'*80}")
            
            results.append(test_result)
    
    # Save results
    log("\n[6/6] Saving results...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'experiments/results/correct_comparison_{timestamp}.json'
    
    os.makedirs('experiments/results', exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'experiment':  'correct_three_way_comparison',
            'timestamp': timestamp,
            'total_tests': len(results),
            'queries': queries,
            'thresholds': price_thresholds,
            'selectivities':  selectivities,
            'cache_stats': cache_stats,
            'results': results
        }, f, indent=2)
    
    log(f"  ✓ Results saved to:  {output_file}")
    
    # Summary statistics
    log("\n" + "=" * 80)
    log("AGGREGATE STATISTICS")
    log("=" * 80)
    
    # Filter out errors
    valid_results = [r for r in results if all([
        r.get('bitmap_time_ms') is not None,
        r.get('post_old_time_ms') is not None,
        r.get('post_fixed_time_ms') is not None
    ])]
    
    log(f"\nValid tests: {len(valid_results)}/{len(results)}")
    
    if valid_results:
        bitmap_times = [r['bitmap_time_ms'] for r in valid_results]
        post_old_times = [r['post_old_time_ms'] for r in valid_results]
        post_fixed_times = [r['post_fixed_time_ms'] for r in valid_results]
        
        log(f"\n{'Method':<20} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10}")
        log(f"{'-'*60}")
        log(f"{'TRUE Bitmap':<20} {np.mean(bitmap_times):>8.2f}ms {np.median(bitmap_times):>8.2f}ms {np.min(bitmap_times):>8.2f}ms {np.max(bitmap_times):>8.2f}ms")
        log(f"{'Post-Filter OLD':<20} {np.mean(post_old_times):>8.2f}ms {np.median(post_old_times):>8.2f}ms {np.min(post_old_times):>8.2f}ms {np.max(post_old_times):>8.2f}ms")
        log(f"{'Post-Filter FIXED':<20} {np.mean(post_fixed_times):>8.2f}ms {np.median(post_fixed_times):>8.2f}ms {np.min(post_fixed_times):>8.2f}ms {np.max(post_fixed_times):>8.2f}ms")
        
        # Recall analysis
        bitmap_recall = np.mean([r['bitmap_results_found'] for r in valid_results]) / 10 * 100
        post_old_recall = np.mean([r['post_old_results_found'] for r in valid_results]) / 10 * 100
        post_fixed_recall = np.mean([r['post_fixed_results_found'] for r in valid_results]) / 10 * 100
        
        log(f"\n{'Method':<20} {'Avg Results':<15} {'Recall %':<10}")
        log(f"{'-'*50}")
        log(f"{'TRUE Bitmap':<20} {np.mean([r['bitmap_results_found'] for r in valid_results]):>6.1f}/10     {bitmap_recall:>8.1f}%")
        log(f"{'Post-Filter OLD':<20} {np.mean([r['post_old_results_found'] for r in valid_results]):>6.1f}/10     {post_old_recall:>8.1f}%")
        log(f"{'Post-Filter FIXED':<20} {np.mean([r['post_fixed_results_found'] for r in valid_results]):>6.1f}/10     {post_fixed_recall:>8.1f}%")
        
        # Breakdown by selectivity
        log(f"\n{'='*80}")
        log("BREAKDOWN BY SELECTIVITY")
        log(f"{'='*80}")
        
        selectivity_ranges = [
            (0, 1, "Ultra-Low (<1%)"),
            (1, 5, "Low (1-5%)"),
            (5, 15, "Medium (5-15%)"),
            (15, 100, "High (>15%)")
        ]
        
        for min_sel, max_sel, label in selectivity_ranges:
            range_results = [r for r in valid_results if min_sel <= r['selectivity_pct'] < max_sel]
            
            if range_results:
                log(f"\n{label} - {len(range_results)} tests")
                log(f"{'─'*60}")
                
                bitmap_avg = np.mean([r['bitmap_time_ms'] for r in range_results])
                post_old_avg = np.mean([r['post_old_time_ms'] for r in range_results])
                post_fixed_avg = np.mean([r['post_fixed_time_ms'] for r in range_results])
                
                bitmap_recall_range = np.mean([r['bitmap_results_found'] for r in range_results]) / 10 * 100
                post_old_recall_range = np.mean([r['post_old_results_found'] for r in range_results]) / 10 * 100
                post_fixed_recall_range = np.mean([r['post_fixed_results_found'] for r in range_results]) / 10 * 100
                
                log(f"  TRUE Bitmap:    {bitmap_avg:>7.2f}ms (recall: {bitmap_recall_range:>5.1f}%)")
                log(f"  Post-OLD:      {post_old_avg:>7.2f}ms (recall: {post_old_recall_range:>5.1f}%)")
                log(f"  Post-FIXED:     {post_fixed_avg:>7.2f}ms (recall: {post_fixed_recall_range:>5.1f}%)")
    
    log("\n" + "=" * 80)
    log("EXPERIMENT COMPLETE!")
    log("=" * 80)
    log(f"\nResults file: {output_file}")

if __name__ == '__main__':
    run_correct_comparison()