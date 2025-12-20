import numpy as np
from src.engine import ThesisEngine
import time
import json
from datetime import datetime
from typing import List, Dict, Tuple

def compute_distance(vec1, vec2):
    """Compute L2 distance between two vectors."""
    return float(np.linalg.norm(vec1 - vec2))

def get_item_details(engine, item_id):
    """Get human-readable details for an item."""
    if item_id < 0 or item_id >= len(engine.vectors):
        return {"id": item_id, "title": "INVALID", "price": 0}
    
    title = f"Item #{item_id}"
    if hasattr(engine, 'raw_items') and item_id < len(engine.raw_items):
        title = engine.raw_items[item_id].get('title', f'Item #{item_id}')[:60]
    
    return {
        "id": int(item_id),
        "title": title,
        "price": float(engine.item_prices[item_id]),
    }

def verify_filter(engine, item_id, filter_func):
    """Check if item passes the filter."""
    try:
        return filter_func(item_id)
    except: 
        return False

def format_results_table(engine, query_vec, results_with_distances, filter_func, method_name):
    """Format results as a nice table with validation."""
    lines = []
    lines.append(f"\n  {method_name} Results (k={len(results_with_distances)}):")
    lines.append(f"  {'Rank':<6} | {'Item ID':<8} | {'Distance':<10} | {'Price':<10} | {'Title':<50} | {'Filter':<6}")
    lines.append(f"  {'-'*6}|{'-'*10}|{'-'*12}|{'-'*12}|{'-'*52}|{'-'*7}")
    
    distances = []
    pass_count = 0
    
    for rank, (item_id, distance) in enumerate(results_with_distances, 1):
        details = get_item_details(engine, item_id)
        passes_filter = verify_filter(engine, item_id, filter_func)
        
        filter_mark = "✅" if passes_filter else "❌"
        if passes_filter:
            pass_count += 1
        
        distances.append(distance)
        
        lines.append(f"  {rank:<6} | {details['id']:<8} | {distance:<10.4f} | ${details['price']:<9.2f} | {details['title']:<50} | {filter_mark:<6}")
    
    avg_distance = np.mean(distances) if distances else 0
    lines.append(f"\n  Avg distance: {avg_distance:.4f}")
    lines.append(f"  Filter pass rate: {pass_count}/{len(results_with_distances)} {'✅' if pass_count == len(results_with_distances) else '❌'}")
    
    return "\n".join(lines), avg_distance, pass_count

def compare_result_sets(true_results, post_results):
    """Compare two result sets."""
    true_ids = set([item_id for item_id, _ in true_results])
    post_ids = set([item_id for item_id, _ in post_results])
    
    common = true_ids.intersection(post_ids)
    only_true = true_ids - post_ids
    only_post = post_ids - true_ids
    
    overlap_pct = len(common) / max(len(true_ids), len(post_ids)) * 100 if true_ids or post_ids else 0
    
    return {
        "overlap_count": len(common),
        "overlap_pct": overlap_pct,
        "only_in_true": list(only_true),
        "only_in_post": list(only_post),
        "common_ids": list(common),
    }

def search_with_distances(engine, query_vec, method, filter_func, filter_name, field=None, threshold=None, k=10):
    """
    Call search method and return results with distances.
    Returns:  (item_ids, distances, time_ms, detailed_timing)
    """
    if method == "true_bitmap":
        if field is None or threshold is None:
            return None, None, None, None
        
        start = time.perf_counter()
        item_ids, search_time = engine.search_bitmap_filter_global_hnsw(
            query_vec, field, threshold, k=k
        )
        total_time = search_time * 1000
        
        # Compute distances for returned items
        distances = []
        for item_id in item_ids:
            if item_id < len(engine.vectors):
                dist = compute_distance(query_vec, engine.vectors[item_id])
                distances.append(dist)
            else:
                distances.append(float('inf'))
        
        return item_ids, distances, total_time, None
    
    elif method == "post_filter": 
        start = time.perf_counter()
        item_ids, search_time = engine.search_post_filter_hnsw_numeric(
            query_vec, filter_func, filter_name, k=k
        )
        total_time = search_time * 1000
        
        # Compute distances
        distances = []
        for item_id in item_ids: 
            if item_id < len(engine.vectors):
                dist = compute_distance(query_vec, engine.vectors[item_id])
                distances.append(dist)
            else:
                distances.append(float('inf'))
        
        return item_ids, distances, total_time, None
    
    return None, None, None, None

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"detailed_comparison_{timestamp}.log"
    results_file = f"detailed_comparison_{timestamp}.json"
    
    def log(message):
        """Log to both console and file"""
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')
    
    log("="*140)
    log("DETAILED COMPARISON:   TRUE Bitmap vs Post-Filter (With Distance Scores)")
    log("="*140)
    log(f"\nTimestamp: {datetime.now()}")
    log(f"Log file: {log_file}")
    log(f"Results file: {results_file}")
    log("="*140)
    
    # Initialize
    log("\n[1/4] Initializing engine...")
    engine = ThesisEngine()
    engine.load_data('data/meta_Electronics.json')
    
    # Test queries
    test_queries = [
        "laptop computer high quality",
        "wireless bluetooth headphones",
        "gaming mouse RGB",
    ]
    
    log(f"\n[2/4] Test queries: {len(test_queries)}")
    for i, q in enumerate(test_queries, 1):
        log(f"  {i}. '{q}'")
    
    # Define test cases
    log("\n[3/4] Defining test cases...")
    
    test_cases = [
        {"name": "price > $2000", "type": "numeric", "threshold": 2000, "field": "price",
         "filter_func": lambda nid: engine.item_prices[nid] > 2000},
        
        {"name": "price > $500", "type": "numeric", "threshold": 500, "field":  "price",
         "filter_func": lambda nid: engine.item_prices[nid] > 500},
        
        {"name": "price > $200", "type": "numeric", "threshold": 200, "field":  "price",
         "filter_func": lambda nid: engine.item_prices[nid] > 200},
        
        {"name": "price > $100", "type": "numeric", "threshold": 100, "field": "price",
         "filter_func": lambda nid:  engine.item_prices[nid] > 100},
        
        {"name": "price > $50", "type": "numeric", "threshold": 50, "field": "price",
         "filter_func": lambda nid:  engine.item_prices[nid] > 50},
        
        {"name": "price > $25", "type": "numeric", "threshold": 25, "field":  "price",
         "filter_func": lambda nid: engine.item_prices[nid] > 25},
    ]
    
    log(f"\nTotal test cases: {len(test_cases)}")
    
    log("\n[4/4] Running experiments...")
    log("="*140)
    
    all_results = []
    
    for query_num, query_text in enumerate(test_queries, 1):
        log(f"\n{'='*140}")
        log(f"QUERY {query_num}/{len(test_queries)}: '{query_text}'")
        log(f"{'='*140}")
        
        query_vec = engine.model.encode([query_text])[0].astype('float32')
        
        for test_num, test in enumerate(test_cases, 1):
            log(f"\n{'-'*140}")
            log(f"TEST {test_num}/{len(test_cases)}: {test['name']}")
            log(f"{'-'*140}")
            
            filter_func = test['filter_func']
            
            # Count matching items
            matching_count = sum(1 for i in range(min(100000, len(engine.vectors))) if filter_func(i))
            selectivity = matching_count / min(100000, len(engine.vectors)) * 100
            
            log(f"  Filter: {test['name']}")
            log(f"  Type: {test['type']}")
            log(f"  Matching items (sampled): ~{matching_count:,} ({selectivity:.2f}% selectivity)")
            
            # === STRATEGY 1: TRUE BITMAP ===
            log(f"\n  >>> STRATEGY 1: TRUE Bitmap Filter (Global HNSW)")
            
            true_ids, true_distances, true_time, _ = search_with_distances(
                engine, query_vec, "true_bitmap", filter_func, test['name'],
                field=test['field'], threshold=test['threshold'], k=10
            )
            
            if true_ids is not None and len(true_ids) > 0:
                true_results = list(zip(true_ids, true_distances))
                true_table, true_avg_dist, true_pass_count = format_results_table(
                    engine, query_vec, true_results, filter_func, "TRUE Bitmap"
                )
                log(true_table)
                log(f"  Time: {true_time:.3f}ms")
            else:
                log(f"  No results or error")
                true_results = []
                true_avg_dist = None
                true_pass_count = 0
                true_time = None
            
            # === STRATEGY 2: POST-FILTER ===
            log(f"\n  >>> STRATEGY 2: Post-Filter HNSW")
            
            post_ids, post_distances, post_time, _ = search_with_distances(
                engine, query_vec, "post_filter", filter_func, test['name'], k=10
            )
            
            if post_ids is not None and len(post_ids) > 0:
                post_results = list(zip(post_ids, post_distances))
                post_table, post_avg_dist, post_pass_count = format_results_table(
                    engine, query_vec, post_results, filter_func, "Post-Filter"
                )
                log(post_table)
                log(f"  Time: {post_time:.3f}ms")
            else:
                log(f"  No results or error")
                post_results = []
                post_avg_dist = None
                post_pass_count = 0
                post_time = None
            
            # === COMPARISON ===
            log(f"\n  {'─'*136}")
            log(f"  DETAILED COMPARISON:")
            
            if true_results and post_results:
                comparison = compare_result_sets(true_results, post_results)
                
                log(f"    Common items: {comparison['overlap_count']}/10 ({comparison['overlap_pct']:.1f}% overlap)")
                log(f"    TRUE Bitmap avg distance: {true_avg_dist:.4f}")
                log(f"    Post-Filter avg distance:   {post_avg_dist:.4f}")
                
                if true_avg_dist < post_avg_dist:
                    diff_pct = (post_avg_dist - true_avg_dist) / true_avg_dist * 100
                    log(f"    ✅ TRUE Bitmap found closer items ({diff_pct:.1f}% better)")
                elif post_avg_dist < true_avg_dist:
                    diff_pct = (true_avg_dist - post_avg_dist) / post_avg_dist * 100
                    log(f"    ✅ Post-Filter found closer items ({diff_pct:.1f}% better)")
                else: 
                    log(f"    ⚖️  Same average distance")
                
                if comparison['only_in_true']: 
                    log(f"    Items only in TRUE Bitmap: {comparison['only_in_true']}")
                if comparison['only_in_post']:
                    log(f"    Items only in Post-Filter: {comparison['only_in_post']}")
                
                log(f"\n    Filter correctness:")
                log(f"      TRUE Bitmap: {'✅' if true_pass_count == len(true_results) else '❌'} {true_pass_count}/{len(true_results)} pass filter")
                log(f"      Post-Filter: {'✅' if post_pass_count == len(post_results) else '❌'} {post_pass_count}/{len(post_results)} pass filter")
                
                log(f"\n    Performance:")
                log(f"      TRUE Bitmap:   {true_time:8.3f}ms")
                log(f"      Post-Filter:   {post_time:8.3f}ms")
                
                if true_time < post_time:
                    speedup = post_time / true_time
                    log(f"      Winner: TRUE Bitmap ({speedup:.2f}x faster)")
                    winner = "TRUE Bitmap"
                else:
                    speedup = true_time / post_time
                    log(f"      Winner:  Post-Filter ({speedup:.2f}x faster)")
                    winner = "Post-Filter"
            else:
                log(f"    Cannot compare (insufficient results)")
                winner = "N/A"
                speedup = None
            
            log(f"  {'─'*136}")
            
            # Store results
            result = {
                "query": query_text,
                "filter_name": test['name'],
                "selectivity": selectivity,
                "true_bitmap":  {
                    "time_ms": true_time,
                    "results": [{"id": int(id), "distance": float(dist)} for id, dist in true_results] if true_results else [],
                    "avg_distance":  true_avg_dist,
                    "filter_pass_count": true_pass_count,
                } if true_results else None,
                "post_filter": {
                    "time_ms": post_time,
                    "results": [{"id": int(id), "distance": float(dist)} for id, dist in post_results] if post_results else [],
                    "avg_distance": post_avg_dist,
                    "filter_pass_count": post_pass_count,
                } if post_results else None,
                "comparison": {
                    "overlap_pct": comparison['overlap_pct'] if true_results and post_results else 0,
                    "winner": winner,
                    "speedup": speedup,
                } if true_results and post_results else None,
            }
            all_results.append(result)
    
    # === SUMMARY ===
    log("\n" + "="*140)
    log("📊 SUMMARY")
    log("="*140)
    
    valid_results = [r for r in all_results if r['true_bitmap'] and r['post_filter']]
    
    if valid_results: 
        log(f"\n{'Query':<30} | {'Filter':<20} | {'Sel%':<6} | {'TRUE(ms)':<10} | {'POST(ms)':<10} | {'Overlap':<8} | {'Winner':<15}")
        log("-"*140)
        for r in valid_results:
            log(f"{r['query'][:28]:<30} | {r['filter_name'][:18]:<20} | {r['selectivity']:>5.2f}% | "
                f"{r['true_bitmap']['time_ms']:>9.3f} | {r['post_filter']['time_ms']:>9.3f} | "
                f"{r['comparison']['overlap_pct']:>6.1f}% | {r['comparison']['winner']:<15}")
    
    log("\n" + "="*140)
    log(f"Experiment complete! Results saved to:")
    log(f"  - Log:    {log_file}")
    log(f"  - JSON:  {results_file}")
    log("="*140)
    
    # Save JSON
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "queries": test_queries,
            "results": all_results,
        }, f, indent=2)

if __name__ == "__main__":
    main()