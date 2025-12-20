import numpy as np
from src.engine import ThesisEngine
import time
import json
from datetime import datetime

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"selectivity_sweep_{timestamp}.log"
    results_file = f"selectivity_sweep_{timestamp}. json"
    
    def log(message):
        """Log to both console and file"""
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')
    
    log("="*120)
    log("SELECTIVITY SWEEP EXPERIMENT:    TRUE Bitmap vs Post-Filter (NO CACHING)")
    log("="*120)
    log(f"\nTimestamp: {datetime.now()}")
    log(f"Log file: {log_file}")
    log(f"Results file: {results_file}")
    
    log("\nHypothesis:")
    log("  - Low selectivity (<1.5%): TRUE Bitmap should win")
    log("  - High selectivity (>1.5%): Post-Filter should win")
    log("  - Goal: Find the exact crossover point")
    log("="*120)
    
    # Initialize
    log("\n[1/3] Initializing engine...")
    engine = ThesisEngine()
    engine.load_data('data/meta_Electronics.json')
    
    # Test query
    query_text = "high quality laptop computer"
    query_vec = engine.model.encode([query_text])[0].astype('float32')
    
    log(f"\n[2/3] Test query: '{query_text}'")
    log(f"      Dataset size: {len(engine.vectors):,} items")
    
    # Use KNOWN thresholds with known selectivities
    log("\n[3/3] Defining test cases (using known thresholds)...")
    
    test_cases = [
        # From our previous experiments + additional points to fill gaps
        {"name": "price > $2000", "threshold": 2000, "expected_sel": 0.05},
        {"name": "price > $1500", "threshold":  1500, "expected_sel": 0.08},
        {"name": "price > $1000", "threshold":  1000, "expected_sel": 0.10},
        {"name": "price > $777", "threshold": 777, "expected_sel": 0.15},
        {"name": "price > $500", "threshold": 500, "expected_sel": 0.25},
        {"name": "price > $400", "threshold": 400, "expected_sel": 0.35},
        {"name": "price > $333. 33", "threshold": 333.33, "expected_sel": 0.51},
        {"name": "price > $300", "threshold": 300, "expected_sel": 0.57},
        {"name": "price > $250", "threshold": 250, "expected_sel": 0.75},
        {"name": "price > $200", "threshold": 200, "expected_sel": 0.92},
        {"name": "price > $175", "threshold": 175, "expected_sel": 1.10},
        {"name": "price > $150", "threshold": 150, "expected_sel": 1.28},
        {"name": "price > $137", "threshold": 137, "expected_sel": 1.43},
        {"name": "price > $125", "threshold": 125, "expected_sel": 1.60},
        {"name": "price > $100", "threshold": 100, "expected_sel": 1.85},
        {"name": "price > $90", "threshold": 90, "expected_sel": 2.10},
        {"name": "price > $75", "threshold": 75, "expected_sel": 2.54},
        {"name": "price > $60", "threshold": 60, "expected_sel": 3.20},
        {"name": "price > $50", "threshold": 50, "expected_sel": 3.68},
        {"name": "price > $40", "threshold": 40, "expected_sel": 4.50},
        {"name": "price > $30", "threshold": 30, "expected_sel": 5.80},
        {"name": "price > $25", "threshold": 25, "expected_sel": 7.06},
        {"name": "price > $20", "threshold": 20, "expected_sel": 9.00},
        {"name": "price > $15", "threshold": 15, "expected_sel": 12.00},
    ]
    
    log(f"\nTest cases:  {len(test_cases)} thresholds")
    log(f"\n{'Test':<25} | {'Threshold':<12} | {'Expected Sel':<15}")
    log("-"*120)
    for tc in test_cases:
        log(f"{tc['name']: <25} | ${tc['threshold']:<11.2f} | {tc['expected_sel']:<14.2f}%")
    
    log("\n" + "="*120)
    log("RUNNING EXPERIMENTS (TRUE BITMAP VS POST-FILTER)")
    log("="*120)
    
    all_results = []
    
    for test_num, test in enumerate(test_cases, 1):
        threshold = test['threshold']
        
        log(f"\n{'='*120}")
        log(f"TEST {test_num}/{len(test_cases)}: {test['name']}")
        log(f"{'='*120}")
        
        # Create filter function
        filter_func = lambda nid, t=threshold: engine.item_prices[nid] > t
        
        # Count actual matching items
        matching_count = sum(1 for i in range(len(engine.vectors)) if filter_func(i))
        actual_selectivity = matching_count / len(engine.vectors) * 100
        
        log(f"  Expected selectivity: {test['expected_sel']:.2f}%")
        log(f"  Actual selectivity:   {actual_selectivity:.2f}%")
        log(f"  Matching items:        {matching_count: ,}")
        
        # === STRATEGY 1: TRUE BITMAP (GLOBAL HNSW) ===
        log(f"\n  >>> STRATEGY 1: TRUE Bitmap Filter (Global HNSW)")
        
        try:
            true_ids, true_time = engine.search_bitmap_filter_global_hnsw(
                query_vec, 'price', threshold, k=10
            )
            true_time_ms = true_time * 1000
            true_found = len(true_ids)
            true_error = None
            log(f"      Time:     {true_time_ms:.2f}ms")
            log(f"      Results: {true_found}/10")
        except Exception as e:
            true_time_ms = None
            true_found = 0
            true_error = str(e)
            log(f"      ERROR: {e}")
        
        # === STRATEGY 2: POST-FILTER HNSW ===
        log(f"\n  >>> STRATEGY 2: Post-Filter HNSW")
        
        try:
            post_ids, post_time = engine.search_post_filter_hnsw_numeric(
                query_vec, filter_func, test['name'], k=10
            )
            post_time_ms = post_time * 1000
            post_found = len(post_ids)
            post_error = None
            log(f"      Time:     {post_time_ms:. 2f}ms")
            log(f"      Results: {post_found}/10")
        except Exception as e:
            post_time_ms = None
            post_found = 0
            post_error = str(e)
            log(f"      ERROR:  {e}")
        
        # === COMPARISON ===
        log(f"\n  {'─'*110}")
        log(f"  COMPARISON:")
        
        if true_time_ms is not None and post_time_ms is not None:
            log(f"    TRUE Bitmap:   {true_time_ms: 7.2f}ms")
            log(f"    Post-Filter:   {post_time_ms: 7.2f}ms")
            
            if true_time_ms < post_time_ms: 
                winner = "TRUE Bitmap"
                speedup = post_time_ms / true_time_ms
                margin_ms = post_time_ms - true_time_ms
            else:
                winner = "Post-Filter"
                speedup = true_time_ms / post_time_ms
                margin_ms = true_time_ms - post_time_ms
            
            log(f"    Winner:  {winner} ({speedup:.2f}x faster, {margin_ms: +.2f}ms)")
        else:
            winner = "ERROR"
            speedup = None
            margin_ms = None
            log(f"    ERROR in one or both strategies")
        
        log(f"  {'─'*110}")
        
        # Store results
        result = {
            "test_number": test_num,
            "name": test['name'],
            "threshold": threshold,
            "expected_selectivity": test['expected_sel'],
            "actual_selectivity":  actual_selectivity,
            "matching_items": matching_count,
            "true_bitmap_ms": true_time_ms,
            "true_bitmap_found": true_found,
            "true_bitmap_error": true_error,
            "post_filter_ms": post_time_ms,
            "post_filter_found": post_found,
            "post_filter_error": post_error,
            "winner": winner,
            "speedup": speedup,
            "margin_ms": margin_ms,
        }
        all_results.append(result)
    
    # === ANALYSIS ===
    log("\n" + "="*120)
    log("📊 COMPLETE RESULTS TABLE")
    log("="*120)
    
    log(f"\n{'Selectivity':<12} | {'Items':<10} | {'TRUE Bitmap':<14} | {'Post-Filter':<14} | {'Winner':<15} | {'Speedup':<10} | {'Margin':<10}")
    log("-"*120)
    for r in all_results:
        if r['true_bitmap_ms'] is not None and r['post_filter_ms'] is not None:
            log(f"{r['actual_selectivity']:<11.2f}% | {r['matching_items']:<10,} | {r['true_bitmap_ms']:>12.2f}ms | {r['post_filter_ms']:>12.2f}ms | {r['winner']:<15} | {r['speedup']: >8.2f}x | {r['margin_ms']:>+9.2f}ms")
    
    # Find crossover point
    log("\n" + "="*120)
    log("🔍 CROSSOVER POINT ANALYSIS")
    log("="*120)
    
    valid_results = [r for r in all_results if r['true_bitmap_ms'] is not None and r['post_filter_ms'] is not None]
    
    true_wins = [r for r in valid_results if r['winner'] == 'TRUE Bitmap']
    post_wins = [r for r in valid_results if r['winner'] == 'Post-Filter']
    
    log(f"\nTRUE Bitmap wins:   {len(true_wins)}/{len(valid_results)} tests")
    log(f"Post-Filter wins:  {len(post_wins)}/{len(valid_results)} tests")
    
    if true_wins: 
        log(f"\nTRUE Bitmap winning range:")
        log(f"  Selectivity: {min(r['actual_selectivity'] for r in true_wins):.2f}% - {max(r['actual_selectivity'] for r in true_wins):.2f}%")
        log(f"  Avg time: {np.mean([r['true_bitmap_ms'] for r in true_wins]):.2f}ms")
        log(f"  Avg speedup: {np.mean([r['speedup'] for r in true_wins]):.2f}x")
        log(f"  Best speedup: {max(r['speedup'] for r in true_wins):.2f}x at {[r['actual_selectivity'] for r in true_wins if r['speedup'] == max(r['speedup'] for r in true_wins)][0]:.2f}%")
    
    if post_wins:
        log(f"\nPost-Filter winning range:")
        log(f"  Selectivity: {min(r['actual_selectivity'] for r in post_wins):.2f}% - {max(r['actual_selectivity'] for r in post_wins):.2f}%")
        log(f"  Avg time: {np.mean([r['post_filter_ms'] for r in post_wins]):.2f}ms")
        log(f"  Avg speedup: {np.mean([r['speedup'] for r in post_wins]):.2f}x")
        log(f"  Best speedup: {max(r['speedup'] for r in post_wins):.2f}x at {[r['actual_selectivity'] for r in post_wins if r['speedup'] == max(r['speedup'] for r in post_wins)][0]:.2f}%")
    
    # Find exact crossover
    if true_wins and post_wins:
        max_true_win_selectivity = max(r['actual_selectivity'] for r in true_wins)
        min_post_win_selectivity = min(r['actual_selectivity'] for r in post_wins)
        
        log(f"\n🎯 CROSSOVER POINT:")
        log(f"  TRUE Bitmap wins up to:      ~{max_true_win_selectivity:.2f}% selectivity")
        log(f"  Post-Filter wins starting:   ~{min_post_win_selectivity:.2f}% selectivity")
        
        if max_true_win_selectivity < min_post_win_selectivity:
            crossover = (max_true_win_selectivity + min_post_win_selectivity) / 2
            log(f"  Crossover zone:               {max_true_win_selectivity:. 2f}% - {min_post_win_selectivity:.2f}%")
            log(f"  Estimated crossover:         ~{crossover:.2f}%")
        else:
            log(f"  ⚠️  Overlapping wins - no clear crossover")
    
    # Performance by selectivity bucket
    log("\n" + "="*120)
    log("📊 PERFORMANCE BY SELECTIVITY RANGE")
    log("="*120)
    
    buckets = [
        (0, 0.25, "Ultra-low (<0.25%)"),
        (0.25, 0.5, "Very low (0.25-0.5%)"),
        (0.5, 1.0, "Low (0.5-1.0%)"),
        (1.0, 1.5, "Medium-low (1.0-1.5%)"),
        (1.5, 2.5, "Medium (1.5-2.5%)"),
        (2.5, 5.0, "Medium-high (2.5-5%)"),
        (5.0, 10.0, "High (5-10%)"),
        (10.0, 20.0, "Very high (>10%)"),
    ]
    
    for min_sel, max_sel, label in buckets:
        bucket_results = [r for r in valid_results if min_sel <= r['actual_selectivity'] < max_sel]
        
        if bucket_results:
            log(f"\n{label}:")
            log(f"  Tests: {len(bucket_results)}")
            log(f"  TRUE Bitmap avg: {np.mean([r['true_bitmap_ms'] for r in bucket_results]):.2f}ms")
            log(f"  Post-Filter avg: {np.mean([r['post_filter_ms'] for r in bucket_results]):.2f}ms")
            
            bucket_true_wins = sum(1 for r in bucket_results if r['winner'] == 'TRUE Bitmap')
            bucket_post_wins = sum(1 for r in bucket_results if r['winner'] == 'Post-Filter')
            
            log(f"  TRUE Bitmap wins: {bucket_true_wins}/{len(bucket_results)}")
            log(f"  Post-Filter wins: {bucket_post_wins}/{len(bucket_results)}")
            
            if bucket_true_wins > bucket_post_wins:
                log(f"  ✅ TRUE Bitmap dominates this range")
            elif bucket_post_wins > bucket_true_wins:
                log(f"  ✅ Post-Filter dominates this range")
            else:
                log(f"  ⚖️  Competitive in this range")
    
    log("\n" + "="*120)
    log("🎯 FINAL CONCLUSIONS")
    log("="*120)
    
    log("\n1. Strategy Selection Rule:")
    if true_wins and post_wins and max_true_win_selectivity < min_post_win_selectivity:
        crossover = (max_true_win_selectivity + min_post_win_selectivity) / 2
        log(f"   ```python")
        log(f"   if selectivity < {crossover:.1f}%:")
        log(f"       use TRUE_BITMAP_GLOBAL_HNSW  # {np.mean([r['true_bitmap_ms'] for r in true_wins]):.1f}ms avg")
        log(f"   else:")
        log(f"       use POST_FILTER_HNSW  # {np. mean([r['post_filter_ms'] for r in post_wins]):.1f}ms avg")
        log(f"   ```")
    elif len(true_wins) > len(post_wins):
        log(f"   ✅ TRUE Bitmap dominates across all selectivities tested")
    elif len(post_wins) > len(true_wins):
        log(f"   ✅ Post-Filter dominates across all selectivities tested")
    
    log("\n2. Performance Summary:")
    if valid_results:
        log(f"   TRUE Bitmap:")
        log(f"     - Best:   {min(r['true_bitmap_ms'] for r in valid_results):.2f}ms")
        log(f"     - Worst: {max(r['true_bitmap_ms'] for r in valid_results):.2f}ms")
        log(f"     - Range: {max(r['true_bitmap_ms'] for r in valid_results) / min(r['true_bitmap_ms'] for r in valid_results):.1f}x variation")
        log(f"   Post-Filter:")
        log(f"     - Best:  {min(r['post_filter_ms'] for r in valid_results):.2f}ms")
        log(f"     - Worst: {max(r['post_filter_ms'] for r in valid_results):.2f}ms")
        log(f"     - Range: {max(r['post_filter_ms'] for r in valid_results) / min(r['post_filter_ms'] for r in valid_results):.1f}x variation")
    
    log("\n3. Hypothesis Validation:")
    if true_wins and post_wins: 
        log(f"   ✅ CONFIRMED:  Selectivity determines optimal strategy")
        log(f"   ✅ TRUE Bitmap excels at low selectivity (<{max_true_win_selectivity:. 1f}%)")
        log(f"   ✅ Post-Filter excels at high selectivity (>{min_post_win_selectivity:. 1f}%)")
        log(f"   ✅ Crossover point found at ~{crossover:.1f}% selectivity")
    elif len(true_wins) == 0:
        log(f"   ⚠️  Post-Filter wins across ALL selectivities")
    elif len(post_wins) == 0:
        log(f"   ⚠️  TRUE Bitmap wins across ALL selectivities")
    
    log("\n" + "="*120)
    log(f"Experiment complete! Results saved to:")
    log(f"  - Log:   {log_file}")
    log(f"  - JSON: {results_file}")
    log("="*120)
    
    # Save JSON
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "query":  query_text,
            "total_items": len(engine.vectors),
            "results": all_results,
            "summary": {
                "true_bitmap_wins": len(true_wins),
                "post_filter_wins":  len(post_wins),
                "crossover_selectivity": crossover if (true_wins and post_wins and max_true_win_selectivity < min_post_win_selectivity) else None,
            }
        }, f, indent=2)

if __name__ == "__main__": 
    main()