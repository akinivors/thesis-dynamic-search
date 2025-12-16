import sys
import os
import time
import numpy as np
import faiss

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.engine import ThesisEngine

def run_hnsw_battle():
    engine = ThesisEngine()
    engine.load_data("data/meta_Electronics.json") 
    
    # We define 10 cases from Massive -> Tiny
    # (Update these with real counts from your finding script if needed)
    test_cases = [
        {"name": "Computers", "type": "Category", "query": "high performance desktop"},    # ~244k
        {"name": "Camera & Photo", "type": "Category", "query": "dslr camera"},            # ~125k
        {"name": "Cell Phones & Accessories", "type": "Category", "query": "smartphone case"}, # ~64k
        {"name": "Car Electronics", "type": "Category", "query": "car audio"},             # ~23k (Boundary)
        {"name": "Sony", "type": "Brand", "query": "noise cancelling headphones"},         # ~12k
        {"name": "Samsung", "type": "Brand", "query": "galaxy phone"},                     # ~6k
        {"name": "Nakamichi", "type": "Brand", "query": "high quality cable"},             # ~99
        {"name": "Sangean", "type": "Brand", "query": "radio"},                           # ~50
        {"name": "Lensse", "type": "Brand", "query": "camera lens"},                       # ~14
        {"name": "INNODESIGN", "type": "Brand", "query": "headphones"}                     # ~6
    ]
    
    print(f"\n{'='*100}")
    print(f"{'TARGET':<25} | {'COUNT':<8} | {'PRE-FILTER':<12} | {'POST(FLAT)':<12} | {'POST(HNSW)':<12} | {'WINNER'}")
    print(f"{'-'*100}")
    
    for case in test_cases:
        target = case['name']
        q_text = case['query']
        
        # Vectorize
        q_vec = engine.model.encode([q_text])[0].astype('float32')
        faiss.normalize_L2(q_vec.reshape(1, -1))
        
        # 1. Get Count
        count = engine.value_counts.get(target, 0)
        if count == 0: continue # Skip invalid
        
        # 2. Method A: Pre-Filter (The Linear Scan)
        start = time.perf_counter()
        _, _ = engine.search_pre_filter(q_vec, target, k=10)
        t_pre = (time.perf_counter() - start) * 1000
        
        # 3. Method B: Post-Filter FLAT (The Baseline)
        start = time.perf_counter()
        _, _ = engine.search_post_filter(q_vec, target, k=10)
        t_flat = (time.perf_counter() - start) * 1000
        
        # 4. Method C: Post-Filter HNSW (The Challenger)
        start = time.perf_counter()
        _, _ = engine.search_post_filter_hnsw(q_vec, target, k=10)
        t_hnsw = (time.perf_counter() - start) * 1000
        
        # Who won?
        times = {"PRE": t_pre, "FLAT": t_flat, "HNSW": t_hnsw}
        winner = min(times, key=times.get)
        
        print(f"{target:<25} | {count:<8} | {t_pre:>9.2f} ms | {t_flat:>9.2f} ms | {t_hnsw:>9.2f} ms | {winner} 🏆")

if __name__ == "__main__":
    run_hnsw_battle()
