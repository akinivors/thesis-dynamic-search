import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import json
from datetime import datetime
from src.engine import ThesisEngine
import faiss

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def compute_distance_stats(distances):
    """Compute statistics for distance scores"""
    valid_dists = [d for d in distances if d != float('inf') and d != -1 and not np.isnan(d)]
    if not valid_dists:
        return None
    return {
        'mean': float(np.mean(valid_dists)),
        'std': float(np.std(valid_dists)),
        'min': float(np.min(valid_dists)),
        'max': float(np.max(valid_dists)),
        'median': float(np.median(valid_dists))
    }

def get_product_details(engine, item_ids, distances=None):
    """Get full product details for results"""
    products = []
    for idx, item_id in enumerate(item_ids):
        if item_id == -1 or item_id >= len(engine.vectors):
            continue
        
        node_data = engine.graph.nodes[item_id]
        
        product = {
            'item_id': int(item_id),
            'asin': node_data.get('asin', 'N/A'),
            'title': node_data.get('title', 'N/A')[:100],  # Truncate long titles
            'brand': node_data.get('brand', 'Unknown'),
            'category': node_data.get('category', 'Unknown'),
            'price': float(node_data.get('price', 0.0)),
            'rating': float(node_data.get('rating', 0.0))
        }
        
        if distances is not None and idx < len(distances):
            if not np.isnan(distances[idx]) and distances[idx] != float('inf'):
                product['distance'] = float(distances[idx])
                product['similarity_score'] = float(1.0 / (1.0 + distances[idx]))
        
        products.append(product)
    
    return products

def analyze_result_overlap(bitmap_ids, post_old_ids, post_fixed_ids):
    """Analyze which results overlap between methods"""
    set_bitmap = set([i for i in bitmap_ids if i != -1])
    set_old = set([i for i in post_old_ids if i != -1])
    set_fixed = set([i for i in post_fixed_ids if i != -1])
    
    return {
        'all_three': len(set_bitmap & set_old & set_fixed),
        'bitmap_and_old': len(set_bitmap & set_old),
        'bitmap_and_fixed': len(set_bitmap & set_fixed),
        'old_and_fixed': len(set_old & set_fixed),
        'bitmap_only': len(set_bitmap - set_old - set_fixed),
        'old_only': len(set_old - set_bitmap - set_fixed),
        'fixed_only': len(set_fixed - set_bitmap - set_old),
        'total_unique': len(set_bitmap | set_old | set_fixed)
    }

def make_filter_function(engine, filter_type, threshold):
    """Create filter function with proper closure"""
    if filter_type == 'price': 
        def filter_func(idx):
            return engine.item_prices[idx] > threshold
    else:  # rating
        def filter_func(idx):
            return engine.item_ratings[idx] > threshold
    return filter_func

def run_ultra_detailed_comparison():
    log("=" * 100)
    log("ULTRA-DETAILED COMPREHENSIVE COMPARISON EXPERIMENT")
    log("=" * 100)
    
    # Load engine
    log("\n[1/7] Loading engine...")
    engine = ThesisEngine()
    
    log("\n[2/7] Loading data...")
    engine.load_data('data/meta_Electronics.json')
    
    log(f"\n  ✓ Loaded {len(engine.vectors)} items")
    log(f"  ✓ HNSW index built with {engine.hnsw_index.ntotal} vectors")
    
    # EXPANDED test queries - 15 diverse queries
    queries = [
        # High-end products
        "laptop computer high quality",
        "professional camera DSLR",
        "gaming laptop RGB mechanical keyboard",
        
        # Mid-range products
        "wireless bluetooth headphones",
        "gaming mouse RGB",
        "portable external hard drive",
        "tablet computer touchscreen",
        
        # Low-end products
        "USB cable charger",
        "phone case protective",
        "screen protector tempered glass",
        
        # Specific brands/features
        "apple macbook pro",
        "samsung galaxy phone",
        "sony noise cancelling headphones",
        
        # Generic searches
        "computer accessories",
        "audio equipment speakers"
    ]
    
    # EXPANDED filters - price + rating
    test_configs = []
    
    # Price filters (12 thresholds)
    price_thresholds = [10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500]
    for threshold in price_thresholds: 
        test_configs.append({
            'filter_type': 'price',
            'operator': '>',
            'threshold': threshold,
            'name': f"price > ${threshold}"
        })
    
    # Rating filters (5 thresholds)
    rating_thresholds = [2.5, 3.0, 3.5, 4.0, 4.5]
    for threshold in rating_thresholds:
        test_configs.append({
            'filter_type': 'rating',
            'operator': '>',
            'threshold': threshold,
            'name': f"rating > {threshold}"
        })
    
    total_tests = len(queries) * len(test_configs)
    log(f"\n[3/7] Test Configuration:")
    log(f"  Queries: {len(queries)}")
    log(f"  Filters: {len(test_configs)} ({len(price_thresholds)} price + {len(rating_thresholds)} rating)")
    log(f"  Total tests: {total_tests}")
    log(f"  Estimated time: ~{total_tests * 0.05:.0f}s ({total_tests * 0.05 / 60:.1f} minutes)")
    
    # Pre-compute selectivities
    log("\n[4/7] Computing selectivities...")
    selectivity_info = {}
    
    for config in test_configs:
        filter_type = config['filter_type']
        threshold = config['threshold']
        
        if filter_type == 'price': 
            count = int(np.sum(engine.item_prices > threshold))
        else:  # rating
            count = int(np.sum(engine.item_ratings > threshold))
        
        selectivity = (count / len(engine.vectors)) * 100
        
        selectivity_info[config['name']] = {
            'filter_type': filter_type,
            'threshold': threshold,
            'count': count,
            'percentage': round(selectivity, 3)
        }
        
        log(f"  {config['name']:<20}:  {count:>7,} items ({selectivity:>6.2f}%)")
    
    # Show cache stats
    cache_stats = engine.get_cache_stats()
    log(f"\n  Cache Status:")
    log(f"    Price cached: {len([k for k in cache_stats['numeric_cached'] if 'price' in k])}")
    log(f"    Rating cached: {len([k for k in cache_stats['numeric_cached'] if 'rating' in k])}")
    
    # Run experiments
    log("\n[5/7] Running ultra-detailed experiments...")
    log("=" * 100)
    
    results = []
    test_num = 0
    
    for query_text in queries:
        log(f"\n{'='*100}")
        log(f"QUERY: '{query_text}'")
        log(f"{'='*100}")
        
        # Encode query
        query_embedding = engine.model.encode([query_text])[0].astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        for config in test_configs:
            test_num += 1
            
            filter_name = config['name']
            filter_type = config['filter_type']
            threshold = config['threshold']
            
            sel_info = selectivity_info[filter_name]
            
            log(f"\n{'-'*100}")
            log(f"[TEST {test_num}/{total_tests}] {filter_name} | Selectivity: {sel_info['percentage']:.2f}% ({sel_info['count']:,} items)")
            log(f"{'-'*100}")
            
            test_result = {
                'test_num': test_num,
                'query':  query_text,
                'filter':  {
                    'type': filter_type,
                    'operator':  '>',
                    'threshold': threshold,
                    'name': filter_name
                },
                'selectivity': sel_info
            }
            
            # Create filter function with proper closure
            filter_func = make_filter_function(engine, filter_type, threshold)
            
            # Estimate selectivity and k_search values
            sample_size = 1000
            sample_indices = np.random.choice(len(engine.vectors), min(sample_size, len(engine.vectors)), replace=False)
            matching_sample = sum(1 for i in sample_indices if filter_func(i))
            est_selectivity_pct = (matching_sample / len(sample_indices)) * 100
            
            # OLD k_search
            if est_selectivity_pct < 1.0:
                k_search_old = 1000
            elif est_selectivity_pct < 5.0:
                k_search_old = 500
            else:
                k_search_old = max(150, int(10 * 100 / max(est_selectivity_pct, 0.1)))
            k_search_old = min(k_search_old, len(engine.vectors))
            
            # ADAPTIVE k_search
            if est_selectivity_pct < 0.1:
                k_search_adaptive = 50000
                strategy = "ultra-low (<0.1%)"
            elif est_selectivity_pct < 0.5:
                k_search_adaptive = 20000
                strategy = "very-low (0.1-0.5%)"
            elif est_selectivity_pct < 1.0:
                k_search_adaptive = 10000
                strategy = "low (0.5-1%)"
            elif est_selectivity_pct < 2.0:
                k_search_adaptive = 5000
                strategy = "medium-low (1-2%)"
            elif est_selectivity_pct < 5.0:
                k_search_adaptive = 2000
                strategy = "medium (2-5%)"
            elif est_selectivity_pct < 10.0:
                k_search_adaptive = 1000
                strategy = "medium-high (5-10%)"
            else:
                k_search_adaptive = max(300, int(10 * 100 / max(est_selectivity_pct, 0.1)))
                strategy = "high (>10%)"
            k_search_adaptive = min(k_search_adaptive, len(engine.vectors))
            
            # === METHOD 1: TRUE BITMAP ===
            log(f"\n>>> METHOD 1: TRUE Bitmap Filter (Global HNSW)")
            try:
                bitmap_ids, bitmap_time = engine.search_bitmap_filter_global_hnsw(
                    query_embedding, filter_type, threshold, k=10
                )
                
                # Get distances
                if filter_type == 'price':
                    valid_ids = np.where(engine.item_prices > threshold)[0]
                else: 
                    valid_ids = np.where(engine.item_ratings > threshold)[0]
                
                bitmap_distances = []
                if len(valid_ids) > 0 and len(bitmap_ids) > 0:
                    selector = faiss.IDSelectorBatch(np.array(sorted(valid_ids), dtype=np.int64))
                    params = faiss.SearchParametersHNSW()
                    params.sel = selector
                    params.efSearch = 200
                    D, I = engine.hnsw_index.search(query_embedding.reshape(1, -1), min(10, len(valid_ids)), params=params)
                    bitmap_distances = D[0].tolist()
                
                bitmap_products = get_product_details(engine, bitmap_ids, bitmap_distances)
                
                test_result['bitmap'] = {
                    'time_ms': round(bitmap_time * 1000, 3),
                    'results_found': len(bitmap_products),
                    'products': bitmap_products,
                    'distance_stats': compute_distance_stats(bitmap_distances)
                }
                
                if bitmap_products:
                    prices = [p['price'] for p in bitmap_products]
                    ratings = [p['rating'] for p in bitmap_products]
                    test_result['bitmap']['result_stats'] = {
                        'price_mean': round(float(np.mean(prices)), 2),
                        'price_std': round(float(np.std(prices)), 2),
                        'price_min': round(float(np.min(prices)), 2),
                        'price_max': round(float(np.max(prices)), 2),
                        'rating_mean':  round(float(np.mean(ratings)), 2)
                    }
                
            except Exception as e:
                log(f"    ❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
                test_result['bitmap'] = {'error': str(e)}
            
            # === METHOD 2: POST-FILTER OLD ===
            log(f"\n>>> METHOD 2: Post-Filter OLD (Fixed k_search)")
            try:
                post_old_ids, post_old_time = engine.search_post_filter_hnsw_numeric(
                    query_embedding, filter_func, filter_name, k=10, adaptive_k_search=False
                )
                
                # Get distances
                D_all, I_all = engine.hnsw_index.search(query_embedding.reshape(1, -1), k_search_old)
                post_old_distances = []
                for pid in post_old_ids: 
                    if pid != -1 and pid in I_all[0]:
                        idx = np.where(I_all[0] == pid)[0][0]
                        post_old_distances.append(float(D_all[0][idx]))
                    else:
                        post_old_distances.append(float('inf'))
                
                post_old_products = get_product_details(engine, post_old_ids, post_old_distances)
                
                test_result['post_old'] = {
                    'time_ms': round(post_old_time * 1000, 3),
                    'results_found': len(post_old_products),
                    'k_search_used': k_search_old,
                    'estimated_selectivity': round(est_selectivity_pct, 2),
                    'products': post_old_products,
                    'distance_stats':  compute_distance_stats(post_old_distances)
                }
                
                if post_old_products:
                    prices = [p['price'] for p in post_old_products]
                    ratings = [p['rating'] for p in post_old_products]
                    test_result['post_old']['result_stats'] = {
                        'price_mean': round(float(np.mean(prices)), 2),
                        'price_std': round(float(np.std(prices)), 2),
                        'price_min': round(float(np.min(prices)), 2),
                        'price_max':  round(float(np.max(prices)), 2),
                        'rating_mean': round(float(np.mean(ratings)), 2)
                    }
                
            except Exception as e: 
                log(f"    ❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
                test_result['post_old'] = {'error': str(e)}
            
            # === METHOD 3: POST-FILTER ADAPTIVE ===
            log(f"\n>>> METHOD 3: Post-Filter ADAPTIVE")
            try:
                post_adaptive_ids, post_adaptive_time = engine.search_post_filter_hnsw_numeric(
                    query_embedding, filter_func, filter_name, k=10, adaptive_k_search=True
                )
                
                # Get distances
                D_all, I_all = engine.hnsw_index.search(query_embedding.reshape(1, -1), k_search_adaptive)
                post_adaptive_distances = []
                for pid in post_adaptive_ids:
                    if pid != -1 and pid in I_all[0]: 
                        idx = np.where(I_all[0] == pid)[0][0]
                        post_adaptive_distances.append(float(D_all[0][idx]))
                    else: 
                        post_adaptive_distances.append(float('inf'))
                
                post_adaptive_products = get_product_details(engine, post_adaptive_ids, post_adaptive_distances)
                
                test_result['post_adaptive'] = {
                    'time_ms': round(post_adaptive_time * 1000, 3),
                    'results_found': len(post_adaptive_products),
                    'k_search_used': k_search_adaptive,
                    'k_search_strategy': strategy,
                    'estimated_selectivity': round(est_selectivity_pct, 2),
                    'products': post_adaptive_products,
                    'distance_stats': compute_distance_stats(post_adaptive_distances)
                }
                
                if post_adaptive_products: 
                    prices = [p['price'] for p in post_adaptive_products]
                    ratings = [p['rating'] for p in post_adaptive_products]
                    test_result['post_adaptive']['result_stats'] = {
                        'price_mean': round(float(np.mean(prices)), 2),
                        'price_std': round(float(np.std(prices)), 2),
                        'price_min':  round(float(np.min(prices)), 2),
                        'price_max': round(float(np.max(prices)), 2),
                        'rating_mean': round(float(np.mean(ratings)), 2)
                    }
                
            except Exception as e:
                log(f"    ❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
                test_result['post_adaptive'] = {'error':  str(e)}
            
            # === OVERLAP ANALYSIS ===
            try:
                bitmap_ids_list = [p['item_id'] for p in test_result.get('bitmap', {}).get('products', [])]
                old_ids_list = [p['item_id'] for p in test_result.get('post_old', {}).get('products', [])]
                adaptive_ids_list = [p['item_id'] for p in test_result.get('post_adaptive', {}).get('products', [])]
                
                test_result['overlap_analysis'] = analyze_result_overlap(
                    bitmap_ids_list, old_ids_list, adaptive_ids_list
                )
            except: 
                pass
            
            # Quick summary
            log(f"\n{'-'*100}")
            log(f"RESULTS SUMMARY:")
            
            if 'bitmap' in test_result and 'time_ms' in test_result['bitmap']:
                log(f"  TRUE Bitmap:      {test_result['bitmap']['time_ms']:>8.2f}ms  |  Found: {test_result['bitmap']['results_found']}/10")
            
            if 'post_old' in test_result and 'time_ms' in test_result['post_old']:
                log(f"  Post-OLD:       {test_result['post_old']['time_ms']:>8.2f}ms  |  Found: {test_result['post_old']['results_found']}/10  |  k={test_result['post_old']['k_search_used']}")
            
            if 'post_adaptive' in test_result and 'time_ms' in test_result['post_adaptive']: 
                log(f"  Post-ADAPTIVE:    {test_result['post_adaptive']['time_ms']:>8.2f}ms  |  Found: {test_result['post_adaptive']['results_found']}/10  |  k={test_result['post_adaptive']['k_search_used']} ({test_result['post_adaptive']['k_search_strategy']})")
            
            if 'overlap_analysis' in test_result:
                log(f"  Overlap:          All 3: {test_result['overlap_analysis']['all_three']}, Unique: {test_result['overlap_analysis']['total_unique']}")
            
            log(f"{'-'*100}")
            
            results.append(test_result)
    
    # Save results
    log("\n[6/7] Saving ultra-detailed results...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'experiments/results/ultra_detailed_{timestamp}.json'
    
    os.makedirs('experiments/results', exist_ok=True)
    
    final_output = {
        'experiment':  'ultra_detailed_comparison',
        'timestamp': timestamp,
        'configuration': {
            'total_tests': len(results),
            'num_queries': len(queries),
            'num_filters': len(test_configs),
            'queries': queries,
            'filters': test_configs
        },
        'selectivity_info': selectivity_info,
        'cache_stats': cache_stats,
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    log(f"  ✓ Results saved to:  {output_file}")
    log(f"  ✓ File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    log("\n" + "=" * 100)
    log("ULTRA-DETAILED EXPERIMENT COMPLETE!")
    log("=" * 100)

if __name__ == '__main__':
    run_ultra_detailed_comparison()