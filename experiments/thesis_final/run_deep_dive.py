import sys
import os
import time
import numpy as np
import faiss
import json

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(current_dir, 'src')):
    project_root = current_dir
else:
    project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.engine import ThesisEngine

def calculate_recall(ground_truth_ids, candidate_ids):
    """Calculate recall:  what fraction of ground truth items were found? """
    if not ground_truth_ids or not candidate_ids:
        return 0.0
    
    gt_set = set(ground_truth_ids)
    cand_set = set(candidate_ids)
    found = len(gt_set.intersection(cand_set))
    return found / len(gt_set)

def get_product_details(engine, node_ids):
    """Extract product details for a list of node IDs"""
    products = []
    for nid in node_ids:
        if nid in engine.graph.nodes:
            node_data = engine.graph.nodes[nid]
            products.append({
                "node_id": int(nid),
                "asin": node_data.get('asin', 'N/A'),
                "title": node_data.get('title', 'N/A')[: 100],
                "brand": node_data.get('brand', 'N/A'),
                "category": node_data.get('category', 'N/A')
            })
    return products

def run_matrix_deep_dive():
    print("="*120)
    print("   [System] Initializing Engine & Loading FULL Data...")
    print("="*120)
    
    try:
        engine = ThesisEngine()
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize engine:  {e}")
        return
    
    data_path = os.path.join(project_root, "data", "meta_Electronics.json")
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Data file not found at {data_path}")
        return
    
    try:
        engine.load_data(data_path)
    except Exception as e: 
        print(f"❌ ERROR: Failed to load data:  {e}")
        return
    
    if engine.vectors is None or len(engine.vectors) == 0:
        print("❌ ERROR: No vectors loaded!")
        return
    
    print(f"✅ Data loaded successfully:  {len(engine.vectors)} items")
    
    # --- WARM UP ---
    print("\n   [System] Warming up.. .", end="", flush=True)
    try:
        dummy_vec = np.random.rand(384).astype('float32')
        faiss.normalize_L2(dummy_vec. reshape(1, -1))
        engine.search_pre_filter(dummy_vec, "Sony", k=5)
        print(" Done.")
    except Exception as e: 
        print(f"\n⚠️ WARNING: Warm-up failed: {e}")

    # TEST MATRIX
    test_matrix = [
        {
            "target":  "Computers",
            "type": "Category",
            "queries": [
                "high performance desktop",
                "gaming laptop 16gb ram",
                "cheap office pc"
            ]
        },
        {
            "target": "Cell Phones & Accessories",
            "type": "Category",
            "queries": [
                "smartphone case shockproof",
                "iphone charger cable",
                "screen protector samsung"
            ]
        },
        {
            "target": "Sony",
            "type": "Brand",
            "queries": [
                "noise cancelling headphones",
                "digital camera dslr",
                "wireless bluetooth speaker"
            ]
        },
        {
            "target":  "Samsung",
            "type": "Brand",
            "queries": [
                "galaxy smartphone",
                "solid state drive ssd",
                "android tablet"
            ]
        },
        {
            "target":  "Nakamichi",
            "type": "Brand",
            "queries": [
                "high quality rca cable",
                "gold plated banana plugs",
                "car audio speaker connector"
            ]
        },
        {
            "target":  "INNODESIGN",
            "type": "Brand",
            "queries": [
                "headphones",
                "earphones",
                "stylish headset"
            ]
        }
    ]
    
    k_values = [10, 50]
    full_report = []
    detailed_results = []
    build_times = {}  # Track build times separately

    print(f"\n{'='*120}")
    print(f"STARTING MATRIX EXPERIMENT (6 Targets x 3 Queries x 2 K-Values x 3 Strategies)")
    print(f"Note: Build times tracked separately and excluded from averages")
    print(f"{'='*120}")

    total_experiments = len(test_matrix) * len(k_values) * 3
    completed = 0

    for k in k_values:
        print(f"\n>>> K={k} BLOCK")
        
        for group in test_matrix:
            target = group['target']
            count = engine.value_counts.get(target, 0)
            
            if count == 0:
                print(f"   ⚠️ WARNING: Target '{target}' not found!  Skipping...")
                continue
            
            print(f"   Target: {target} ({count} items)")
            
            for query_text in group['queries']:
                completed += 1
                print(f"      Q{completed}/{total_experiments}:  '{query_text}'")
                
                try:
                    # Vectorize
                    q_vec = engine.model.encode([query_text])[0].astype('float32')
                    faiss.normalize_L2(q_vec.reshape(1, -1))
                    
                    # Ground truth
                    gt_ids, gt_time = engine.search_pre_filter(q_vec, target, k=k)
                    
                    result_row = {
                        "k":  k,
                        "target":  target,
                        "count": count,
                        "query": query_text,
                        "strategies": {}
                    }
                    
                    detailed_row = {
                        "k":  k,
                        "target":  target,
                        "query": query_text,
                        "ground_truth": get_product_details(engine, gt_ids),
                        "strategies": {}
                    }
                    
                    # Test strategies
                    strategies = [
                        ("Pre-Filter Flat", "flat"),
                        ("Pre-Filter HNSW", "hnsw_pre"),
                        ("Post-Filter HNSW", "hnsw_post")
                    ]
                    
                    for name, method_type in strategies:
                        try:
                            if method_type == "flat":
                                found_ids, method_time = engine.search_pre_filter(q_vec, target, k=k)
                                build_time_ms = 0
                            elif method_type == "hnsw_pre":
                                found_ids, method_time, build_time = engine.search_pre_filter_hnsw(q_vec, target, k=k)
                                build_time_ms = build_time * 1000
                                # Track build time separately
                                if build_time_ms > 0:
                                    if target not in build_times:
                                        build_times[target] = build_time_ms
                            else:  # hnsw_post
                                found_ids, method_time = engine.search_post_filter_hnsw(q_vec, target, k=k)
                                build_time_ms = 0
                            
                            duration = method_time * 1000  # Search time only
                            recall = calculate_recall(gt_ids, found_ids)
                            
                            result_row["strategies"][name] = {
                                "time_ms": duration,
                                "recall": recall,
                                "num_results": len(found_ids),
                                "build_time_ms": build_time_ms
                            }
                            
                            detailed_row["strategies"][name] = {
                                "time_ms": duration,
                                "recall": recall,
                                "build_time_ms": build_time_ms,
                                "products": get_product_details(engine, found_ids)
                            }
                            
                            recall_icon = "✅" if recall >= 0.95 else "⚠️" if recall >= 0.8 else "❌"
                            
                            # Display with build time if applicable
                            if build_time_ms > 0:
                                print(f"         - {name: <18} | {duration: >7. 2f}ms | Recall: {recall*100:>5.1f}% {recall_icon} [+{build_time_ms:. 0f}ms build]")
                            else: 
                                print(f"         - {name:<18} | {duration:>7.2f}ms | Recall: {recall*100:>5.1f}% {recall_icon}")
                            
                        except Exception as e:
                            print(f"         - {name:<18} | ERROR: {str(e)[:50]}")
                            result_row["strategies"][name] = {"error": str(e), "time_ms": 0, "recall": 0}
                            detailed_row["strategies"][name] = {"error": str(e), "products": []}
                    
                    full_report.append(result_row)
                    detailed_results.append(detailed_row)
                    
                except Exception as e:
                    print(f"         ❌ ERROR: {e}")
                    continue

    # SAVE RESULTS
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Summary
    summary_path = os. path.join(results_dir, "matrix_deep_dive_fixed.json")
    with open(summary_path, "w") as f:
        json.dump(full_report, f, indent=4)
    
    # Detailed products
    detailed_path = os. path.join(results_dir, "matrix_detailed_products_fixed.json")
    with open(detailed_path, "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    # Build times summary
    build_times_path = os.path.join(results_dir, "hnsw_build_times. json")
    with open(build_times_path, "w") as f:
        json.dump(build_times, f, indent=4)
    
    print(f"\n{'='*120}")
    print(f"✅ Experiment Complete!")
    print(f"📂 Summary: {summary_path}")
    print(f"📂 Detailed: {detailed_path}")
    print(f"📂 Build Times: {build_times_path}")
    print(f"📊 Total:  {len(full_report)} experiments")
    print(f"{'='*120}")
    
    # Summary stats (EXCLUDING build times)
    if full_report:
        print("\n📈 QUICK SUMMARY (Search Time Only, Build Times Excluded):")
        for strategy_name in ["Pre-Filter Flat", "Pre-Filter HNSW", "Post-Filter HNSW"]: 
            times = [r["strategies"][strategy_name]["time_ms"]
                    for r in full_report
                    if strategy_name in r["strategies"] and "time_ms" in r["strategies"][strategy_name]]
            recalls = [r["strategies"][strategy_name]["recall"]
                      for r in full_report
                      if strategy_name in r["strategies"] and "recall" in r["strategies"][strategy_name]]
            
            if times and recalls:
                avg_time = np.mean(times)
                avg_recall = np.mean(recalls)
                print(f"   {strategy_name: <18}:  Avg Time={avg_time: >6.2f}ms, Avg Recall={avg_recall*100:>5.1f}%")
        
        # Show build times separately
        if build_times: 
            print("\n🔨 HNSW BUILD TIMES (One-Time Cost Per Attribute):")
            for attr, build_ms in sorted(build_times.items(), key=lambda x: x[1], reverse=True):
                count = engine.value_counts.get(attr, 0)
                print(f"   {attr: <30}: {build_ms:>7.0f}ms ({count: >6} items)")

if __name__ == "__main__":
    try:
        run_matrix_deep_dive()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted!")
    except Exception as e:
        print(f"\n\n❌ FATAL:  {e}")
        import traceback
        traceback.print_exc()