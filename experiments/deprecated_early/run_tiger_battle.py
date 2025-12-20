import sys
import os
import time
import numpy as np
import faiss

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.engine import ThesisEngine

def run_tiger_battle():
    print("   [System] Initializing Engine & Loading FULL Data...")
    engine = ThesisEngine()
    
    # REMOVED LIMIT: We are running the full battle now!
    engine.load_data("data/meta_Electronics.json") 
    
    # 11 Data Points covering the full spectrum
    test_cases = [
        # --- MASSIVE (The "HNSW Post" Zone) ---
        {"name": "Computers", "type": "Category", "query": "high performance desktop"},      # ~244k
        {"name": "Camera & Photo", "type": "Category", "query": "dslr camera"},              # ~125k
        {"name": "Home Audio & Theater", "type": "Category", "query": "surround sound"},     # ~102k (New)
        
        # --- HIGH (The "Battleground") ---
        {"name": "Cell Phones & Accessories", "type": "Category", "query": "smartphone case"}, # ~64k
        {"name": "Car Electronics", "type": "Category", "query": "car audio"},               # ~23k
        
        # --- MEDIUM (The "HNSW Bitmap" Zone) ---
        {"name": "Sony", "type": "Brand", "query": "noise cancelling headphones"},           # ~12k
        {"name": "Generic", "type": "Brand", "query": "usb cable"},                          # ~11.5k (New)
        {"name": "Dell", "type": "Brand", "query": "laptop charger"},                        # ~7.5k (New)
        {"name": "Samsung", "type": "Brand", "query": "galaxy phone"},                       # ~6.7k
        
        # --- RARE (The "Flat Scan" Zone) ---
        {"name": "Nakamichi", "type": "Brand", "query": "high quality cable"},               # ~99
        {"name": "Lensse", "type": "Brand", "query": "camera lens"},                         # ~14 (Restored)
        {"name": "INNODESIGN", "type": "Brand", "query": "headphones"}                       # ~6
    ]
    
    print(f"\n{'='*145}")
    print(f"{'TARGET':<25} | {'COUNT':<8} | {'FLAT PRE':<12} | {'FLAT POST':<12} | {'HNSW POST':<12} | {'HNSW BITMAP':<12} | {'WINNER'}")
    print(f"{'-'*145}")
    
    for case in test_cases:
        target = case['name']
        q_text = case['query']
        
        # Vectorize
        q_vec = engine.model.encode([q_text])[0].astype('float32')
        faiss.normalize_L2(q_vec.reshape(1, -1))
        
        # Get Count
        count = engine.value_counts.get(target, 0)
        if count == 0: continue
        
        # 1. FLAT PRE-FILTER (Linear Scan)
        start = time.perf_counter()
        engine.search_pre_filter(q_vec, target, k=10)
        t_flat_pre = (time.perf_counter() - start) * 1000
        
        # 2. FLAT POST-FILTER (Baseline Index)
        start = time.perf_counter()
        engine.search_post_filter(q_vec, target, k=10)
        t_flat_post = (time.perf_counter() - start) * 1000
        
        # 3. HNSW POST-FILTER (Naive Graph)
        start = time.perf_counter()
        engine.search_post_filter_hnsw(q_vec, target, k=10)
        t_hnsw_post = (time.perf_counter() - start) * 1000
        
        # 4. HNSW BITMAP (TigerVector Approach)
        start = time.perf_counter()
        try:
            engine.search_hnsw_bitmap(q_vec, target, k=10)
            t_hnsw_bitmap = (time.perf_counter() - start) * 1000
        except Exception as e:
            t_hnsw_bitmap = 99999.9 
            # print(f"(Err: {e}) ", end="") 

        # Determine Winner
        times = {
            "Scan": t_flat_pre, 
            "FlatPost": t_flat_post, 
            "HNSWPost": t_hnsw_post, 
            "Bitmap": t_hnsw_bitmap
        }
        winner = min(times, key=times.get)
        
        print(f"{target:<25} | {count:<8} | {t_flat_pre:>9.2f} ms | {t_flat_post:>9.2f} ms | {t_hnsw_post:>9.2f} ms | {t_hnsw_bitmap:>9.2f} ms | {winner} 🏆")

if __name__ == "__main__":
    run_tiger_battle()