import sys
import os
import time
import numpy as np
import faiss

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.engine import ThesisEngine

def run_comparative_proof():
    # 1. Initialize & Calibrate
    engine = ThesisEngine()
    engine.load_data("data/meta_Electronics.json") 
    engine.calibrate()

    # 2. Define Test Cases (INCLUDING NEW EDGE CASES)
    test_cases = [
        {
            "query": "high performance desktop",
            "target": "Computers",
            "tier": "MASSIVE (> 100k)"
        },
        {
            "query": "car audio system",
            "target": "Car Electronics",  # <--- NEW CRITICAL EDGE CASE (23k items)
            "tier": "HIGH (Just above Threshold)"
        },
        {
            "query": "noise cancelling headphones",
            "target": "Sony",
            "tier": "MEDIUM (Just below Threshold)"
        },
        {
            "query": "high quality cable",
            "target": "Nakamichi",
            "tier": "SMALL (< 100)"
        }
    ]

    print(f"\n{'='*90}")
    print("FINAL THESIS VALIDATION: INSIDE THE OPTIMIZER'S MIND")
    print(f"{'='*90}")
    
    # Print the "Constants" the machine learned
    t_item = engine.cost_per_filter_item * 1000 # Convert to ms
    t_index = engine.cost_index_batch * 1000    # Convert to ms
    threshold = int(engine.cost_index_batch / engine.cost_per_filter_item)
    
    print(f"🧠 MACHINE LEARNING CONSTANTS:")
    print(f"   - Cost to scan 1 item (Linear):   {t_item:.6f} ms")
    print(f"   - Cost to query Index (Fixed):    {t_index:.4f} ms")
    print(f"   - CALCULATED THRESHOLD:           {threshold} items")
    print(f"{'='*90}")

    correct_predictions = 0

    for case in test_cases:
        q_text = case['query']
        brand = case['target']
        tier = case['tier']
        
        print(f"\n>>> CASE: {brand} ({tier})")
        
        # Vectorize
        q_vec = engine.model.encode([q_text])[0].astype('float32')
        faiss.normalize_L2(q_vec.reshape(1, -1))
        
        # --- 1. REVEAL THE OPTIMIZER'S CALCULATION ---
        count = engine.value_counts.get(brand, 0)
        
        # Calculate what the engine sees internally
        pred_pre_cost_ms = count * t_item
        pred_post_cost_ms = t_index
        
        print(f"    [Optimizer's Brain]")
        print(f"       1. Count: {count} items")
        print(f"       2. Pre-Filter Math:  {count} * {t_item:.5f}ms = {pred_pre_cost_ms:.2f} ms")
        print(f"       3. Post-Filter Math: Constant Overhead        = {pred_post_cost_ms:.2f} ms")
        
        if pred_pre_cost_ms < pred_post_cost_ms:
            prediction = "PRE-FILTER"
            reason = f"Scanning is faster (Difference: {pred_post_cost_ms - pred_pre_cost_ms:.2f}ms)"
        else:
            prediction = "POST-FILTER"
            reason = f"Indexing is faster (Difference: {pred_pre_cost_ms - pred_post_cost_ms:.2f}ms)"
            
        print(f"       -> DECISION: {prediction} because {reason}")

        # --- 2. RUN ACTUAL EXPERIMENT ---
        
        # A. Force Pre-Filter
        start = time.perf_counter()
        _, _ = engine.search_pre_filter(q_vec, brand, k=10)
        actual_pre_ms = (time.perf_counter() - start) * 1000
        
        # B. Force Post-Filter
        start = time.perf_counter()
        _, _ = engine.search_post_filter(q_vec, brand, k=10)
        actual_post_ms = (time.perf_counter() - start) * 1000
        
        # --- 3. VERDICT ---
        # Determine the True Winner
        if actual_pre_ms < actual_post_ms:
            true_winner = "PRE-FILTER"
        else:
            true_winner = "POST-FILTER"
            
        is_correct = (prediction == true_winner)
        if is_correct: correct_predictions += 1
        
        icon = "✅" if is_correct else "❌"
        print(f"    [Reality Check] Pre: {actual_pre_ms:.2f}ms | Post: {actual_post_ms:.2f}ms")
        print(f"    [Verdict]       {icon} Optimizer was {true_winner == prediction}")

    print(f"\n{'-'*90}")
    print(f"Summary: Optimizer Accuracy {correct_predictions}/{len(test_cases)}")
    print(f"{'-'*90}")

if __name__ == "__main__":
    run_comparative_proof()