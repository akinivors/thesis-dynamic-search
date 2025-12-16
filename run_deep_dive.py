import sys
import os
import time
import numpy as np
import faiss
import json

# --- SMART PATH FIX ---
# This ensures it works whether you run it from the root OR the experiments folder
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(current_dir, 'src')):
    # We are already at the root
    project_root = current_dir
else:
    # We are deep in experiments, go up two levels
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Add project root to system path so we can import src
if project_root not in sys.path:
    sys.path.append(project_root)

from src.engine import ThesisEngine

def calculate_recall(ground_truth_ids, candidate_ids):
    if not ground_truth_ids: return 0.0
    gt_set = set(ground_truth_ids)
    found = len(gt_set.intersection(candidate_ids))
    return found / len(gt_set)

def run_matrix_deep_dive():
    print("   [System] Initializing Engine & Loading FULL Data...")
    engine = ThesisEngine()
    engine.load_data("data/meta_Electronics.json") 
    
    # --- WARM UP ---
    print("   [System] Warming up...", end="", flush=True)
    dummy_vec = np.random.rand(1, 384).astype('float32')
    engine.search_pre_filter(dummy_vec, "Sony", k=5)
    print(" Done.")

    # 1. MATRIX DEFINITION (Targets x Queries)
    test_matrix = [
        {
            "target": "Computers", 
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
            "target": "Samsung", 
            "type": "Brand",
            "queries": [
                "galaxy smartphone",
                "solid state drive ssd",
                "android tablet"
            ]
        },
        {
            "target": "Nakamichi", 
            "type": "Brand",
            "queries": [
                "high quality rca cable",
                "gold plated banana plugs",
                "car audio speaker connector"
            ]
        },
        {
            "target": "INNODESIGN", 
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

    print(f"\n{'='*120}")
    print(f"STARTING MATRIX EXPERIMENT (6 Targets x 3 Queries x 2 K-Values)")
    print(f"{'='*120}")

    for k in k_values:
        print(f"\n>>> K={k} BLOCK")
        
        for group in test_matrix:
            target = group['target']
            count = engine.value_counts.get(target, 0)
            
            print(f"   Target: {target} ({count} items)")
            
            for query_text in group['queries']:
                # Vectorize
                q_vec = engine.model.encode([query_text])[0].astype('float32')
                faiss.normalize_L2(q_vec.reshape(1, -1))
                
                # --- GROUND TRUTH ---
                gt_ids, _ = engine.search_pre_filter(q_vec, target, k=k)
                
                result_row = {
                    "k": k,
                    "target": target,
                    "count": count,
                    "query": query_text,
                    "strategies": {}
                }
                
                # --- COMPETITORS ---
                strategies = [
                    ("Flat Post", engine.search_post_filter),
                    ("HNSW Post", engine.search_post_filter_hnsw),
                    ("HNSW Bitmap", engine.search_hnsw_bitmap)
                ]
                
                print(f"      Q: '{query_text}'")
                
                for name, method in strategies:
                    start = time.perf_counter()
                    try:
                        found_ids, _ = method(q_vec, target, k=k)
                        duration = (time.perf_counter() - start) * 1000
                        recall = calculate_recall(gt_ids, found_ids)
                        
                        result_row["strategies"][name] = {
                            "time_ms": duration,
                            "recall": recall
                        }
                        
                        print(f"         - {name:<12} | {duration:>6.2f}ms | Recall: {recall*100:>3.0f}%")
                        
                    except Exception as e:
                        result_row["strategies"][name] = {"error": str(e)}

                full_report.append(result_row)

    # 3. ROBUST SAVING
    # This now uses the 'project_root' we calculated at the top
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    out_path = os.path.join(results_dir, "matrix_deep_dive.json")
    
    with open(out_path, "w") as f:
        json.dump(full_report, f, indent=4)
    
    print(f"\n{'='*120}")
    print(f"✅ Matrix Experiment Complete.")
    print(f"📂 Data saved to: {out_path}")

if __name__ == "__main__":
    run_matrix_deep_dive()