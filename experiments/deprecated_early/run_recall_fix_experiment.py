import numpy as np
from src.engine import ThesisEngine
import time
import json
from datetime import datetime

def compute_distance(vec1, vec2):
    """Compute L2 distance."""
    return float(np.linalg.norm(vec1 - vec2))

def get_item_details(engine, item_id):
    """Get item details."""
    if item_id < 0 or item_id >= len(engine.vectors):
        return {"id": item_id, "title": "INVALID", "price": 0}
    return {
        "id": int(item_id),
        "title": f"Item #{item_id}",
        "price": float(engine.item_prices[item_id]),
    }

def verify_filter(engine, item_id, filter_func):
    """Check if item passes filter."""
    try:
        return filter_func(item_id)
    except:
        return False

def format_results_table(engine, query_vec, results_with_distances, filter_func, method_name):
    """Format results table."""
    lines = []
    lines.append(f"\n  {method_name} Results (k={len(results_with_distances)}):")
    lines.append(f"  {'Rank':<6} | {'Item ID':<8} | {'Distance':<10} | {'Price':<10} | {'Filter':<6}")
    lines.append(f"  {'-'*6}|{'-'*10}|{'-'*12}|{'-'*12}|{'-'*7}")
    
    distances = []
    pass_count = 0
    
    for rank, (item_id, distance) in enumerate(results_with_distances, 1):
        details = get_item_details(engine, item_id)
        passes_filter = verify_filter(engine, item_id, filter_func)
        filter_mark = "✅" if passes_filter else "❌"
        if passes_filter:
            pass_count += 1
        distances.append(distance)
        lines.append(f"  {rank:<6} | {details['id']:<8} | {distance:<10.4f} | ${details['price']:<9.2f} | {filter_mark:<6}")
    
    avg_distance = np.mean(distances) if distances else 0
    lines.append(f"\n  Avg distance: {avg_distance:.4f}")
    lines.append(f"  Filter pass: {pass_count}/{len(results_with_distances)} ✅" if pass_count == len(results_with_distances) else f"  Filter pass: {pass_count}/{len(results_with_distances)} ❌")
    
    return "\n".join(lines), avg_distance, pass_count

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"recall_fix_{timestamp}.log"
    results_file = f"recall_fix_{timestamp}.json"
    
    def log(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')
    
    log("="*140)
    log("RECALL FIX EXPERIMENT:   Pre-Filter vs Post-Filter (OLD) vs Post-Filter (FIXED)")
    log("="*140)
    log(f"\nTimestamp: {datetime.now()}")
    log("="*140)
    
    # Initialize
    log("\n[1/3] Initializing engine...")
    engine = ThesisEngine()
    engine.load_data('data/meta_Electronics.json')
    
    # Test queries
    test_queries = [
        "laptop computer high quality",
        "wireless bluetooth headphones",
        "gaming mouse RGB",
        "USB cable charger",
        "smartphone case protective",
    ]
    
    log(f"\n[2/3] Test queries: {len(test_queries)}")
    for i, q in enumerate(test_queries, 1):
        log(f"  {i}. '{q}'")
    
    # Define comprehensive selectivity range
    log("\n[3/3] Defining test cases...")
    
    test_cases = [
        {"name": "price > $2000", "threshold": 2000, "expected_sel": 0.03},
        {"name": "price > $1000", "threshold": 1000, "expected_sel": 0.09},
        {"name": "price > $500", "threshold": 500, "expected_sel": 0.25},
        {"name": "price > $300", "threshold": 300, "expected_sel": 0.57},
        {"name": "price > $200", "threshold": 200, "expected_sel": 0.92},
        {"name": "price > $150", "threshold": 150, "expected_sel": 1.28},
        {"name": "price > $100", "threshold": 100, "expected_sel": 1.85},
        {"name": "price > $75", "threshold": 75, "expected_sel": 2.54},
        {"name": "price > $50", "threshold": 50, "expected_sel": 3.68},
        {"name": "price > $40", "threshold": 40, "expected_sel": 4.49},
        {"name": "price > $30", "threshold": 30, "expected_sel": 5.88},
        {"name": "price > $25", "threshold": 25, "expected_sel": 7.06},
        {"name": "price > $20", "threshold": 20, "expected_sel": 8.58},
        {"name": "price > $15", "threshold": 15, "expected_sel": 12.33},
        {"name": "price > $10", "threshold": 10, "expected_sel": 18.00},
        {"name": "price > $5", "threshold": 5, "expected_sel": 25.00},
    ]
    
    log(f"\nTotal:  {len(test_queries)} queries × {len(test_cases)} filters = {len(test_queries) * len(test_cases)} tests")
    
    log("\n" + "="*140)
    log("RUNNING EXPERIMENTS")
    log("="*140)
    
    all_results = []
    
    for query_num, query_text in enumerate(test_queries, 1):
        log(f"\n{'='*140}")
        log(f"QUERY {query_num}/{len(test_queries)}: '{query_text}'")
        log(f"{'='*140}")
        
        query_vec = engine.model.encode([query_text])[0].astype('float32')
        
        for test_num, test in enumerate(test_cases, 1):
            log(f"\n{'-'*140}")
            log(f"TEST {test_num}/{len(test_cases)}: {test['name']} (expected ~{test['expected_sel']}% selectivity)")
            log(f"{'-'*140}")
            
            threshold = test['threshold']
            filter_func = lambda nid, t=threshold: engine.item_prices[nid] > t
            
            # Count actual matching
            matching_count = sum(1 for i in range(min(10000, len(engine.vectors))) if filter_func(i))
            actual_sel = matching_count / min(10000, len(engine.vectors)) * 100
            log(f"  Actual selectivity (sampled): {actual_sel:.2f}%")
            
            # === METHOD 1: PRE-FILTER (SEPARATE HNSW) ===
            log(f"\n  >>> METHOD 1: Pre-Filter (Separate Cached HNSW)")
            pre_start = time.perf_counter()
            pre_ids, pre_time, _ = engine.search_bitmap_pre_filter_hnsw(
                query_vec, 'price', threshold, k=10
            )
            pre_time_ms = pre_time * 1000
            
            pre_distances = [compute_distance(query_vec, engine.vectors[i]) for i in pre_ids if i < len(engine.vectors)]
            pre_results = list(zip(pre_ids, pre_distances))
            pre_table, pre_avg_dist, pre_pass = format_results_table(engine, query_vec, pre_results, filter_func, "Pre-Filter")
            log(pre_table)
            log(f"  Time: {pre_time_ms:.3f}ms")
            
            # === METHOD 2: POST-FILTER OLD (BROKEN) ===
            log(f"\n  >>> METHOD 2: Post-Filter (OLD - Fixed k_search=1000)")
            post_old_ids, post_old_time = engine.search_post_filter_hnsw_numeric(
                query_vec, filter_func, test['name'], k=10, adaptive_k_search=False
            )
            post_old_time_ms = post_old_time * 1000
            
            post_old_distances = [compute_distance(query_vec, engine.vectors[i]) for i in post_old_ids if i < len(engine.vectors)]
            post_old_results = list(zip(post_old_ids, post_old_distances))
            post_old_table, post_old_avg_dist, post_old_pass = format_results_table(engine, query_vec, post_old_results, filter_func, "Post-OLD")
            log(post_old_table)
            log(f"  Time: {post_old_time_ms:.3f}ms")
            
            # === METHOD 3: POST-FILTER FIXED (ADAPTIVE) ===
            log(f"\n  >>> METHOD 3: Post-Filter (FIXED - Adaptive k_search)")
            post_new_ids, post_new_time = engine.search_post_filter_hnsw_numeric(
                query_vec, filter_func, test['name'], k=10, adaptive_k_search=True
            )
            post_new_time_ms = post_new_time * 1000
            
            post_new_distances = [compute_distance(query_vec, engine.vectors[i]) for i in post_new_ids if i < len(engine.vectors)]
            post_new_results = list(zip(post_new_ids, post_new_distances))
            post_new_table, post_new_avg_dist, post_new_pass = format_results_table(engine, query_vec, post_new_results, filter_func, "Post-FIXED")
            log(post_new_table)
            log(f"  Time: {post_new_time_ms:.3f}ms")
            
            # === COMPARISON ===
            log(f"\n  {'─'*136}")
            log(f"  COMPARISON:")
            log(f"    Pre-Filter:         {pre_time_ms:8.3f}ms   ({len(pre_ids)}/10 results, avg dist: {pre_avg_dist:.4f})")
            log(f"    Post-OLD:          {post_old_time_ms:8.3f}ms   ({len(post_old_ids)}/10 results, avg dist: {post_old_avg_dist:.4f})")
            log(f"    Post-FIXED:        {post_new_time_ms:8.3f}ms   ({len(post_new_ids)}/10 results, avg dist: {post_new_avg_dist:.4f})")
            
            # Determine winners
            times = {"Pre-Filter": pre_time_ms, "Post-OLD": post_old_time_ms, "Post-FIXED": post_new_time_ms}
            speed_winner = min(times, key=times.get)
            
            recalls = {"Pre-Filter": len(pre_ids), "Post-OLD": len(post_old_ids), "Post-FIXED": len(post_new_ids)}
            recall_winner = max(recalls, key=recalls.get)
            
            log(f"\n    Speed winner:      {speed_winner}")
            log(f"    Recall winner:    {recall_winner} ({recalls[recall_winner]}/10)")
            log(f"    Post-FIXED vs Pre:   {post_new_time_ms / pre_time_ms:.2f}x slower, {len(post_new_ids)}/{len(pre_ids)} recall")
            log(f"    Post-FIXED vs OLD:   {post_new_time_ms / post_old_time_ms:.2f}x slower, +{len(post_new_ids) - len(post_old_ids)} results")
            log(f"  {'─'*136}")
            
            # Store results
            result = {
                "query": query_text,
                "filter":  test['name'],
                "selectivity": actual_sel,
                "pre_filter":  {"time_ms": pre_time_ms, "found":  len(pre_ids), "avg_dist": pre_avg_dist},
                "post_old": {"time_ms": post_old_time_ms, "found": len(post_old_ids), "avg_dist": post_old_avg_dist},
                "post_fixed": {"time_ms": post_new_time_ms, "found": len(post_new_ids), "avg_dist": post_new_avg_dist},
                "winners": {"speed": speed_winner, "recall": recall_winner},
            }
            all_results.append(result)
    
    # === SUMMARY ===
    log("\n" + "="*140)
    log("📊 SUMMARY")
    log("="*140)
    
    log(f"\n{'Query':<30} | {'Filter':<15} | {'Sel%':<6} | {'Pre(ms)':<9} | {'OLD(ms)':<9} | {'FIX(ms)':<9} | {'Pre#':<5} | {'OLD#':<5} | {'FIX#':<5} | {'Winner':<12}")
    log("-"*140)
    for r in all_results:
        log(f"{r['query'][:28]:<30} | {r['filter'][:13]:<15} | {r['selectivity']:>5.2f}% | "
            f"{r['pre_filter']['time_ms']:>8.2f} | {r['post_old']['time_ms']:>8.2f} | {r['post_fixed']['time_ms']:>8.2f} | "
            f"{r['pre_filter']['found']:>4} | {r['post_old']['found']:>4} | {r['post_fixed']['found']:>4} | {r['winners']['speed'][:10]:<12}")
    
    log("\n" + "="*140)
    log("ANALYSIS")
    log("="*140)
    
    # Recall analysis
    pre_perfect = sum(1 for r in all_results if r['pre_filter']['found'] == 10)
    old_perfect = sum(1 for r in all_results if r['post_old']['found'] == 10)
    fix_perfect = sum(1 for r in all_results if r['post_fixed']['found'] == 10)
    
    log(f"\nRecall (10/10 results):")
    log(f"  Pre-Filter:    {pre_perfect}/{len(all_results)} ({pre_perfect/len(all_results)*100:.1f}%)")
    log(f"  Post-OLD:      {old_perfect}/{len(all_results)} ({old_perfect/len(all_results)*100:.1f}%)")
    log(f"  Post-FIXED:    {fix_perfect}/{len(all_results)} ({fix_perfect/len(all_results)*100:.1f}%)")
    
    # Speed analysis
    log(f"\nAverage Speed:")
    log(f"  Pre-Filter:    {np.mean([r['pre_filter']['time_ms'] for r in all_results]):.2f}ms")
    log(f"  Post-OLD:      {np.mean([r['post_old']['time_ms'] for r in all_results]):.2f}ms")
    log(f"  Post-FIXED:    {np.mean([r['post_fixed']['time_ms'] for r in all_results]):.2f}ms")
    
    log("\n" + "="*140)
    log(f"Complete!  Saved to {log_file}")
    log("="*140)
    
    # Save JSON
    with open(results_file, 'w') as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": all_results}, f, indent=2)

if __name__ == "__main__":
    main()