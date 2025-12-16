import sys
import os
import time
import numpy as np
import faiss
import json

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.engine import ThesisEngine

def get_product_titles(engine, node_ids):
    """Helper to convert IDs to readable Product Titles"""
    titles = []
    for nid in node_ids:
        # Get title, default to ID if missing
        t = engine.graph.nodes[nid].get('title', f"Item #{nid}")
        # Truncate for cleaner JSON
        titles.append(t[:60] + "..." if len(t) > 60 else t)
    return titles

def calculate_recall(ground_truth_ids, candidate_ids):
    """Calculates how many Ground Truth items were found (0.0 to 1.0)"""
    if not ground_truth_ids: return 0.0
    gt_set = set(ground_truth_ids)
    # Intersection count
    found = len(gt_set.intersection(candidate_ids))
    return found / len(gt_set)

def run_quality_check():
    print("   [System] Initializing Engine & Loading FULL Data...")
    engine = ThesisEngine()
    engine.load_data("data/meta_Electronics.json") 
    
    test_cases = [
        # --- MASSIVE (The "HNSW Post" Zone) ---
        {"name": "Computers", "type": "Category", "query": "high performance desktop"},      
        {"name": "Camera & Photo", "type": "Category", "query": "dslr camera"},              
        
        # --- HIGH (The "Battleground") ---
        {"name": "Cell Phones & Accessories", "type": "Category", "query": "smartphone case"}, 
        {"name": "Car Electronics", "type": "Category", "query": "car audio"},               
        
        # --- MEDIUM (The "HNSW Bitmap" Zone) ---
        {"name": "Sony", "type": "Brand", "query": "noise cancelling headphones"},           
        {"name": "Samsung", "type": "Brand", "query": "galaxy phone"},                       
        
        # --- RARE (The "Flat Scan" Zone) ---
        {"name": "Nakamichi", "type": "Brand", "query": "high quality cable"},               
        {"name": "INNODESIGN", "type": "Brand", "query": "headphones"}                       
    ]
    
    full_results = []
    
    print(f"\n{'='*120}")
    print(f"{'TARGET':<20} | {'METHOD':<15} | {'TIME (ms)':<10} | {'RECALL %':<10} | {'TOP RESULT (Sample)'}")
    print(f"{'-'*120}")
    
    for case in test_cases:
        target = case['name']
        q_text = case['query']
        print(f">>> CASE: {target} ({engine.value_counts.get(target, 0)} items)")
        
        # Vectorize
        q_vec = engine.model.encode([q_text])[0].astype('float32')
        faiss.normalize_L2(q_vec.reshape(1, -1))
        
        # --- 1. GOLD STANDARD: FLAT PRE-FILTER (Scan) ---
        # We assume this is 100% correct because it checks every single item.
        start = time.perf_counter()
        gt_ids, _ = engine.search_pre_filter(q_vec, target, k=10)
        t_gt = (time.perf_counter() - start) * 1000
        gt_titles = get_product_titles(engine, gt_ids)
        
        # Log Ground Truth
        print(f"{'':<20} | {'Flat Scan (GT)':<15} | {t_gt:>9.2f}  | {'100%':<10} | {gt_titles[0] if gt_titles else 'None'}")
        
        case_data = {
            "target": target,
            "query": q_text,
            "ground_truth_time_ms": t_gt,
            "ground_truth_products": gt_titles,
            "strategies": {}
        }

        # --- 2. COMPETITORS ---
        strategies = [
            ("Flat Post", engine.search_post_filter),
            ("HNSW Post", engine.search_post_filter_hnsw),
            ("HNSW Bitmap", engine.search_hnsw_bitmap)
        ]

        for name, method in strategies:
            start = time.perf_counter()
            try:
                # Run Search
                found_ids, _ = method(q_vec, target, k=10)
                duration = (time.perf_counter() - start) * 1000
                
                # Get Quality Metrics
                recall = calculate_recall(gt_ids, found_ids)
                titles = get_product_titles(engine, found_ids)
                
                # Print Row
                print(f"{'':<20} | {name:<15} | {duration:>9.2f}  | {recall*100:>5.0f}%     | {titles[0] if titles else 'None'}")
                
                # Save to Data
                case_data["strategies"][name] = {
                    "time_ms": duration,
                    "recall": recall,
                    "products": titles
                }
                
            except Exception as e:
                print(f"{'':<20} | {name:<15} | {'ERROR':>9}  | {'0%':<10} | {str(e)}")

        print("-" * 120)
        full_results.append(case_data)

    # Save Results to JSON for your Thesis Analysis
    os.makedirs("results", exist_ok=True)
    with open("results/tiger_quality_analysis.json", "w") as f:
        json.dump(full_results, f, indent=4)
    
    print(f"\n✅ Full Quality Analysis saved to: results/tiger_quality_analysis.json")

if __name__ == "__main__":
    run_quality_check()