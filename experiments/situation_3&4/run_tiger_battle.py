import sys
import os
import time
import numpy as np
import faiss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.engine import ThesisEngine

def run_tiger_battle():
    engine = ThesisEngine()
    engine.load_data("data/meta_Electronics.json") 
    
    # We test across the spectrum to find where Bitmap fails
    test_cases = [
        {"name": "Computers", "type": "Category", "query": "high performance desktop"},    # ~244k (Massive)
        {"name": "Cell Phones & Accessories", "type": "Category", "query": "smartphone case"}, # ~64k (High)
        {"name": "Car Electronics", "type": "Category", "query": "car audio"},             # ~23k (Medium)
        {"name": "Sony", "type": "Brand", "query": "noise cancelling headphones"},         # ~12k (Medium-Low)
        {"name": "Samsung", "type": "Brand", "query": "galaxy phone"},                     # ~6k (Low)
        {"name": "Nakamichi", "type": "Brand", "query": "high quality cable"},             # ~99 (Tiny)
        {"name": "INNODESIGN", "type": "Brand", "query": "headphones"}                     # ~6 (Micro)
    ]
    
    print(f"\n{'='*110}")
    print(f"{'TARGET':<25} | {'COUNT':<8} | {'FLAT PRE (Scan)':<15} | {'HNSW POST':<15} | {'HNSW BITMAP':<15} | {'WINNER'}")
    print(f"{'-'*110}")
    
    for case in test_cases:
        target = case['name']
        q_text = case['query']
        q_vec = engine.model.encode([q_text])[0].astype('float32')
        faiss.normalize_L2(q_vec.reshape(1, -1))
        
        count = engine.value_counts.get(target, 0)
        if count == 0: continue
        
        # 1. FLAT PRE-FILTER (Brute Force Scan)
        start = time.perf_counter()
        engine.search_pre_filter(q_vec, target, k=10)
        t_flat_pre = (time.perf_counter() - start) * 1000
        
        # 2. HNSW POST-FILTER (Naive)
        start = time.perf_counter()
        engine.search_post_filter_hnsw(q_vec, target, k=10)
        t_hnsw_post = (time.perf_counter() - start) * 1000
        
        # 3. HNSW BITMAP (TigerVector Standard)
        start = time.perf_counter()
        try:
            engine.search_hnsw_bitmap(q_vec, target, k=10)
            t_hnsw_bitmap = (time.perf_counter() - start) * 1000
        except Exception as e:
            t_hnsw_bitmap = 99999.9 # Fail safe if FAISS version doesn't support selector
            print(f" (Bitmap Error: {e})", end="")

        # Determine Winner
        times = {"Scan": t_flat_pre, "Post": t_hnsw_post, "Bitmap": t_hnsw_bitmap}
        winner = min(times, key=times.get)
        
        print(f"{target:<25} | {count:<8} | {t_flat_pre:>12.2f} ms | {t_hnsw_post:>12.2f} ms | {t_hnsw_bitmap:>12.2f} ms | {winner} 🏆")

if __name__ == "__main__":
    run_tiger_battle()
