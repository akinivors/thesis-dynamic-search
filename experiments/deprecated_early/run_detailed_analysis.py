import numpy as np
from src.engine import ThesisEngine
import time
import json
from datetime import datetime
from typing import List, Dict, Any

def compute_ground_truth(engine, query_vec, filter_func, k=10):
    """
    Compute ground truth by exhaustive search on filtered items. 
    """
    start = time.perf_counter()
    
    # Get all items that pass filter
    valid_indices = [i for i in range(len(engine.vectors)) if filter_func(i)]
    
    if len(valid_indices) == 0:
        return [], (time.perf_counter() - start) * 1000, []
    
    # Compute distances for all valid items
    valid_vectors = engine.vectors[valid_indices]
    distances = np.linalg.norm(valid_vectors - query_vec, axis=1)
    
    # Sort by distance
    sorted_idx = np.argsort(distances)[:k]
    ground_truth_ids = [valid_indices[i] for i in sorted_idx]
    ground_truth_distances = [distances[i] for i in sorted_idx]
    
    compute_time = (time.perf_counter() - start) * 1000
    
    return ground_truth_ids, compute_time, ground_truth_distances

def get_item_details(engine, item_id):
    """Get human-readable details for an item."""
    if item_id < 0 or item_id >= len(engine.vectors):
        return {"id": item_id, "title": "INVALID", "price": 0, "brand": "N/A", "categories": []}
    
    # Get title from raw data
    title = "Unknown"
    if hasattr(engine, 'raw_items') and item_id < len(engine.raw_items):
        title = engine.raw_items[item_id]. get('title', 'Unknown')
    
    # Get price (array)
    price = float(engine.item_prices[item_id]) if item_id < len(engine.item_prices) else 0.0
    
    # Get brand (check if it's array or dict)
    brand = "Unknown"
    if hasattr(engine, 'item_brands'):
        if isinstance(engine.item_brands, dict):
            brand = engine. item_brands.get(item_id, "Unknown")
        elif isinstance(engine.item_brands, np.ndarray):
            brand = str(engine.item_brands[item_id]) if item_id < len(engine.item_brands) else "Unknown"
        else:
            try:
                brand = str(engine.item_brands[item_id])
            except: 
                brand = "Unknown"
    
    # Get categories (check type)
    categories = []
    if hasattr(engine, 'item_categories'):
        if isinstance(engine.item_categories, dict):
            categories = engine.item_categories.get(item_id, [])
        elif isinstance(engine.item_categories, np.ndarray):
            cat = engine.item_categories[item_id] if item_id < len(engine.item_categories) else []
            categories = [cat] if cat else []
    
    # Truncate title
    if len(title) > 80:
        title = title[: 77] + "..."
    
    return {
        "id":  int(item_id),
        "title": title,
        "price": price,
        "brand": brand,
        "categories":  categories if isinstance(categories, list) else [categories] if categories else []
    }

def calculate_recall(ground_truth_ids, retrieved_ids):
    """Calculate recall:  how many ground truth items were retrieved."""
    if len(ground_truth_ids) == 0:
        return 0.0
    
    ground_truth_set = set(ground_truth_ids)
    retrieved_set = set(retrieved_ids)
    intersection = ground_truth_set.intersection(retrieved_set)
    
    return len(intersection) / len(ground_truth_ids)

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"detailed_analysis_{timestamp}.log"
    results_file = f"detailed_analysis_{timestamp}.json"
    
    def log(message):
        """Log to both console and file"""
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')
    
    log("="*140)
    log("DETAILED ANALYSIS:  TRUE Bitmap vs Post-Filter (With Ground Truth & Recall)")
    log("="*140)
    log(f"\nTimestamp: {datetime.now()}")
    log(f"Log file: {log_file}")
    log(f"Results file: {results_file}")
    log("="*140)
    
    # Initialize
    log("\n[1/4] Initializing engine...")
    engine = ThesisEngine()
    engine.load_data('data/meta_Electronics.json')
    
    # Test queries (different types)
    test_queries = [
        "laptop computer high quality",
        "wireless bluetooth headphones",
        "gaming mouse RGB",
        "phone charger cable usb",
        "camera digital photography"
    ]
    
    log(f"\n[2/4] Test queries: {len(test_queries)}")
    for i, q in enumerate(test_queries, 1):
        log(f"  {i}. '{q}'")
    
    # Define comprehensive test cases
    log("\n[3/4] Defining test cases...")
    
    test_cases = [
        # === NUMERIC FILTERS (Various selectivities) ===
        {"name": "price > $2000", "type": "numeric", "threshold": 2000, "field": "price",
         "filter_func": lambda nid: engine.item_prices[nid] > 2000},
        
        {"name": "price > $500", "type": "numeric", "threshold": 500, "field":  "price",
         "filter_func": lambda nid: engine.item_prices[nid] > 500},
        
        {"name": "price > $200", "type": "numeric", "threshold": 200, "field":  "price",
         "filter_func": lambda nid: engine.item_prices[nid] > 200},
        
        {"name": "price > $100", "type": "numeric", "threshold": 100, "field": "price",
         "filter_func": lambda nid: engine.item_prices[nid] > 100},
        
        {"name": "price > $50", "type": "numeric", "threshold": 50, "field": "price",
         "filter_func": lambda nid: engine.item_prices[nid] > 50},
        
        {"name": "price > $25", "type": "numeric", "threshold": 25, "field": "price",
         "filter_func": lambda nid: engine.item_prices[nid] > 25},
        
        # === PRICE RANGES ===
        {"name": "price $50-$100", "type": "numeric_range", "threshold": None, "field": "price",
         "filter_func": lambda nid: 50 < engine.item_prices[nid] <= 100},
        
        {"name": "price $100-$200", "type": "numeric_range", "threshold": None, "field": "price",
         "filter_func": lambda nid:  100 < engine.item_prices[nid] <= 200},
        
        # === CATEGORICAL FILTERS ===
        {"name": "category = Electronics", "type": "categorical", "threshold": None, "field": "category",
         "filter_func": lambda nid: "Electronics" in engine.inverted_index and nid in engine.inverted_index.get("Electronics", set())},
        
        {"name": "category = Computers", "type": "categorical", "threshold": None, "field": "category",
         "filter_func": lambda nid: "Computers" in engine.inverted_index and nid in engine.inverted_index.get("Computers", set())},
    ]
    
    log(f"\nTotal test cases: {len(test_cases)}")
    for i, tc in enumerate(test_cases, 1):
        log(f"  {i}. {tc['name']} ({tc['type']})")
    
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
            count_start = time.perf_counter()
            matching_count = sum(1 for i in range(len(engine.vectors)) if filter_func(i))
            count_time = (time.perf_counter() - count_start) * 1000
            
            selectivity = matching_count / len(engine.vectors) * 100
            
            log(f"  Filter: {test['name']}")
            log(f"  Type: {test['type']}")
            log(f"  Matching items: {matching_count:,} ({selectivity:.2f}% selectivity)")
            log(f"  Count time: {count_time:.3f}ms")
            
            # === GROUND TRUTH ===
            log(f"\n  >>> GROUND TRUTH (Exhaustive Search)")
            gt_ids, gt_time, gt_distances = compute_ground_truth(engine, query_vec, filter_func, k=10)
            log(f"      Total time: {gt_time:.3f}ms")
            log(f"      Found:  {len(gt_ids)}/10 items")
            
            if len(gt_ids) > 0:
                log(f"\n      Ground Truth Results:")
                for rank, item_id in enumerate(gt_ids[:5], 1):  # Show top 5
                    details = get_item_details(engine, item_id)
                    log(f"        {rank}. [{details['id']}] ${details['price']:.2f} - {details['title']}")
            
            # === STRATEGY 1: TRUE BITMAP ===
            log(f"\n  >>> STRATEGY 1: TRUE Bitmap Filter (Global HNSW)")
            
            true_ids = []
            true_time = None
            true_breakdown = {}
            
            if test['type'] in ['numeric'] and test['threshold'] is not None:
                try:
                    # Call with detailed timing
                    true_start = time.perf_counter()
                    true_ids, true_time = engine.search_bitmap_filter_global_hnsw(
                        query_vec, test['field'], test['threshold'], k=10
                    )
                    true_time_ms = true_time * 1000
                    
                    log(f"      Total time: {true_time_ms:.3f}ms")
                    log(f"      Found:  {len(true_ids)}/10 items")
                    
                    if len(true_ids) > 0:
                        log(f"\n      TRUE Bitmap Results:")
                        for rank, item_id in enumerate(true_ids[:5], 1):
                            details = get_item_details(engine, item_id)
                            log(f"        {rank}. [{details['id']}] ${details['price']:.2f} - {details['title']}")
                    
                except Exception as e:
                    log(f"      ERROR: {e}")
                    true_time_ms = None
            else:
                log(f"      SKIPPED (filter type '{test['type']}' not supported)")
                true_time_ms = None
            
            # === STRATEGY 2: POST-FILTER ===
            log(f"\n  >>> STRATEGY 2: Post-Filter HNSW")
            
            try:
                post_start = time.perf_counter()
                post_ids, post_time = engine.search_post_filter_hnsw_numeric(
                    query_vec, filter_func, test['name'], k=10
                )
                post_time_ms = post_time * 1000
                
                log(f"      Total time: {post_time_ms:.3f}ms")
                log(f"      Found: {len(post_ids)}/10 items")
                
                if len(post_ids) > 0:
                    log(f"\n      Post-Filter Results:")
                    for rank, item_id in enumerate(post_ids[: 5], 1):
                        details = get_item_details(engine, item_id)
                        log(f"        {rank}. [{details['id']}] ${details['price']:.2f} - {details['title']}")
                
            except Exception as e:
                log(f"      ERROR: {e}")
                post_time_ms = None
                post_ids = []
            
            # === RECALL CALCULATION ===
            log(f"\n  >>> RECALL ANALYSIS")
            
            if len(gt_ids) > 0:
                if true_time_ms is not None and len(true_ids) > 0:
                    true_recall = calculate_recall(gt_ids, true_ids)
                    log(f"      TRUE Bitmap Recall: {true_recall*100:.1f}% ({len(set(gt_ids).intersection(set(true_ids)))}/{len(gt_ids)} matches)")
                else:
                    true_recall = None
                    log(f"      TRUE Bitmap Recall: N/A")
                
                if post_time_ms is not None and len(post_ids) > 0:
                    post_recall = calculate_recall(gt_ids, post_ids)
                    log(f"      Post-Filter Recall: {post_recall*100:.1f}% ({len(set(gt_ids).intersection(set(post_ids)))}/{len(gt_ids)} matches)")
                else:
                    post_recall = None
                    log(f"      Post-Filter Recall:  N/A")
            else:
                true_recall = None
                post_recall = None
                log(f"      No ground truth items to compare")
            
            # === COMPARISON ===
            log(f"\n  {'─'*136}")
            log(f"  COMPARISON:")
            
            if true_time_ms is not None and post_time_ms is not None:
                log(f"    TRUE Bitmap:    {true_time_ms:8.3f}ms  (Recall: {true_recall*100 if true_recall is not None else 0:.1f}%)")
                log(f"    Post-Filter:    {post_time_ms:8.3f}ms  (Recall: {post_recall*100 if post_recall is not None else 0:.1f}%)")
                log(f"    Ground Truth:  {gt_time:8.3f}ms  (Baseline)")
                
                if true_time_ms < post_time_ms:
                    winner = "TRUE Bitmap"
                    speedup = post_time_ms / true_time_ms
                else:
                    winner = "Post-Filter"
                    speedup = true_time_ms / post_time_ms
                
                log(f"    Winner: {winner} ({speedup:.2f}x faster)")
            else:
                log(f"    Cannot compare (one or both strategies failed)")
                winner = "N/A"
                speedup = None
            
            log(f"  {'─'*136}")
            
            # Store results
            result = {
                "query": query_text,
                "query_num": query_num,
                "test_num": test_num,
                "filter_name": test['name'],
                "filter_type": test['type'],
                "matching_items": matching_count,
                "selectivity": selectivity,
                "ground_truth_time_ms": gt_time,
                "ground_truth_ids": gt_ids,
                "true_bitmap_time_ms": true_time_ms,
                "true_bitmap_ids": true_ids,
                "true_bitmap_recall": true_recall,
                "post_filter_time_ms": post_time_ms,
                "post_filter_ids": post_ids,
                "post_filter_recall":  post_recall,
                "winner": winner,
                "speedup": speedup,
            }
            all_results.append(result)
    
    # === FINAL SUMMARY ===
    log("\n" + "="*140)
    log("📊 COMPLETE RESULTS SUMMARY")
    log("="*140)
    
    valid_results = [r for r in all_results if r['true_bitmap_time_ms'] is not None and r['post_filter_time_ms'] is not None]
    
    log(f"\n{'Query':<30} | {'Filter':<25} | {'Sel%':<6} | {'TRUE(ms)':<10} | {'POST(ms)':<10} | {'T. Recall':<9} | {'P.Recall':<9} | {'Winner':<15}")
    log("-"*140)
    for r in valid_results:
        log(f"{r['query'][:28]:<30} | {r['filter_name'][:23]:<25} | {r['selectivity']:>5.2f}% | {r['true_bitmap_time_ms']:>9.3f} | {r['post_filter_time_ms']:>9.3f} | {(r['true_bitmap_recall'] or 0)*100:>7.1f}% | {(r['post_filter_recall'] or 0)*100:>7.1f}% | {r['winner']:<15}")
    
    # === RECALL ANALYSIS ===
    log("\n" + "="*140)
    log("🎯 RECALL ANALYSIS")
    log("="*140)
    
    true_recalls = [r['true_bitmap_recall'] for r in valid_results if r['true_bitmap_recall'] is not None]
    post_recalls = [r['post_filter_recall'] for r in valid_results if r['post_filter_recall'] is not None]
    
    if true_recalls:
        log(f"\nTRUE Bitmap Recall:")
        log(f"  Average: {np.mean(true_recalls)*100:.1f}%")
        log(f"  Min:  {np.min(true_recalls)*100:.1f}%")
        log(f"  Max: {np.max(true_recalls)*100:.1f}%")
        log(f"  Perfect (100%): {sum(1 for r in true_recalls if r == 1.0)}/{len(true_recalls)} tests")
    
    if post_recalls:
        log(f"\nPost-Filter Recall:")
        log(f"  Average: {np.mean(post_recalls)*100:.1f}%")
        log(f"  Min: {np.min(post_recalls)*100:.1f}%")
        log(f"  Max: {np.max(post_recalls)*100:.1f}%")
        log(f"  Perfect (100%): {sum(1 for r in post_recalls if r == 1.0)}/{len(post_recalls)} tests")
    
    # === PERFORMANCE BY SELECTIVITY ===
    log("\n" + "="*140)
    log("📊 PERFORMANCE BY SELECTIVITY BUCKET")
    log("="*140)
    
    buckets = [
        (0, 0.5, "Ultra-low (<0.5%)"),
        (0.5, 1.0, "Very low (0.5-1.0%)"),
        (1.0, 2.0, "Low (1.0-2.0%)"),
        (2.0, 4.0, "Medium (2.0-4.0%)"),
        (4.0, 8.0, "High (4.0-8.0%)"),
        (8.0, 100.0, "Very high (>8.0%)"),
    ]
    
    for min_sel, max_sel, label in buckets:
        bucket_results = [r for r in valid_results if min_sel <= r['selectivity'] < max_sel]
        
        if bucket_results:
            log(f"\n{label}:")
            log(f"  Tests: {len(bucket_results)}")
            log(f"  TRUE Bitmap avg: {np.mean([r['true_bitmap_time_ms'] for r in bucket_results]):.3f}ms")
            log(f"  Post-Filter avg:  {np.mean([r['post_filter_time_ms'] for r in bucket_results]):.3f}ms")
            
            true_wins = sum(1 for r in bucket_results if r['winner'] == 'TRUE Bitmap')
            log(f"  TRUE Bitmap wins: {true_wins}/{len(bucket_results)}")
            
            if bucket_results[0].get('true_bitmap_recall') is not None:
                avg_true_recall = np.mean([r['true_bitmap_recall'] for r in bucket_results if r['true_bitmap_recall'] is not None])
                avg_post_recall = np.mean([r['post_filter_recall'] for r in bucket_results if r['post_filter_recall'] is not None])
                log(f"  Avg TRUE Recall: {avg_true_recall*100:.1f}%")
                log(f"  Avg POST Recall: {avg_post_recall*100:.1f}%")
    
    log("\n" + "="*140)
    log(f"Experiment complete! Results saved to:")
    log(f"  - Log:   {log_file}")
    log(f"  - JSON: {results_file}")
    log("="*140)
    
        # Save JSON
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    # Convert all results to serializable format
    serializable_results = convert_to_serializable(all_results)
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "queries": test_queries,
            "test_cases": [{"name": t['name'], "type": t['type']} for t in test_cases],
            "results": serializable_results,
            "summary":  {
                "total_tests": len(all_results),
                "valid_comparisons": len(valid_results),
                "true_bitmap_wins": int(sum(1 for r in valid_results if r['winner'] == 'TRUE Bitmap')),
                "post_filter_wins": int(sum(1 for r in valid_results if r['winner'] == 'Post-Filter')),
            }
        }, f, indent=2)

if __name__ == "__main__": 
    main()