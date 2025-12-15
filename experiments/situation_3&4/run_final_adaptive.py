import numpy as np
import faiss
from src.engine import ThesisEngine

def run_adaptive_proof():
    # 1. Initialize System
    engine = ThesisEngine()
    engine.load_data("meta_Electronics.json") # Full Dataset

    # 2. Define The Test Cases (Based on your previous findings)
    # We use the exact same queries and brands to verify the fix.
    
    test_cases = [
        {
            "query": "noise cancelling headphones",
            "target": "Sony",
            "tier": "Common (High Count)",
            "expected": "POST-FILTER" # Pre-filter was 0.73s, Post was 0.02s
        },
        {
            "query": "noise cancelling headphones",
            "target": "INNODESIGN",
            "tier": "Rare (Low Count)",
            "expected": "PRE-FILTER" # Post-filter failed recall
        },
        {
            "query": "gaming laptop high performance",
            "target": "ADAMANT COMPUTERS",
            "tier": "Mid-Tier (Specialized)",
            "expected": "PRE-FILTER" # Post-filter recall good, but Pre-filter was 200x faster
        },
        {
            "query": "dslr camera lens",
            "target": "Lensse",
            "tier": "Rare (Low Count)",
            "expected": "PRE-FILTER" # Post-filter failed recall
        }
    ]

    print(f"\n{'='*60}")
    print("FINAL THESIS VERIFICATION: ADAPTIVE OPTIMIZER")
    print(f"{'='*60}")

    for case in test_cases:
        q_text = case['query']
        brand = case['target']
        tier = case['tier']
        expected = case['expected']
        
        print(f"\n>>> CASE: {brand} ({tier})")
        print(f"    Query: '{q_text}'")
        
        # Vectorize
        q_vec = engine.model.encode([q_text])[0].astype('float32')
        faiss.normalize_L2(q_vec.reshape(1, -1))
        
        # --- ADAPTIVE EXECUTION ---
        # 1. Get Statistics
        # (In a real DB, this is a metadata lookup)
        # We simulate the cost model check:
        if brand in engine.brand_to_ids:
            count = len(engine.brand_to_ids[brand])
        else:
            count = 0
            
        # 2. The Decision Logic (Cost Model)
        # Your data showed Sony (225 items) was slow on Pre-Filter.
        # Adamant (25 items) was fast on Pre-Filter.
        # Let's set the switch point at 100 items.
        THRESHOLD = 100 
        
        if count < THRESHOLD:
            strategy = "PRE-FILTER"
            res, duration = engine.search_pre_filter(q_vec, brand)
        else:
            strategy = "POST-FILTER"
            res, duration = engine.search_post_filter(q_vec, brand)
            
        # 3. Validation
        status = "SUCCESS" if strategy == expected else "FAIL"
        
        print(f"    [Optimizer] Count: {count} | Threshold: {THRESHOLD}")
        print(f"    [Decision]  Chose: {strategy} (Expected: {expected}) -> {status}")
        print(f"    [Result]    Found: {len(res)} items in {duration:.5f}s")
        
        if status == "FAIL":
            print("    !! CRITICAL: Optimizer logic needs tuning.")

if __name__ == "__main__":
    run_adaptive_proof()