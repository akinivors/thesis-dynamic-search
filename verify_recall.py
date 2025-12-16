import sys
import os
import numpy as np
import faiss

# Adjust path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.engine import ThesisEngine

def test_recall_cliff():
    print("Initializing Engine...")
    engine = ThesisEngine()
    engine.load_data("data/meta_Electronics.json") # Ensure this is the full file

    # 1. Setup a Difficult Query
    query_text = "dslr camera lens"
    target_brand = "Lensse" # Known rare brand
    k_target = 10
    
    print(f"\n--- TESTING RECALL FOR RARE BRAND: '{target_brand}' ---")
    print(f"Query: '{query_text}'")
    print(f"Target K: {k_target}")

    # Vectorize
    q_vec = engine.model.encode([query_text])[0].astype('float32')
    faiss.normalize_L2(q_vec.reshape(1, -1))

    # 2. Run Post-Filter
    print("\nRunning Iterative Post-Filter...")
    results, duration = engine.search_post_filter(q_vec, target_brand, k=k_target)
    
    # 3. Verify
    print(f"Found: {len(results)} items")
    print(f"Time:  {duration:.4f}s")
    
    details = engine.get_details(results)
    for i, d in enumerate(details):
        print(f"  {i+1}. {d}")

    # 4. Success Condition
    if len(results) >= k_target:
        print("\n✅ PASSED: Retrieved full K results despite sparsity.")
    elif len(results) > 0:
        print("\n⚠️ WARNING: Found some results, but not K. (Brand might be too small?)")
    else:
        print("\n❌ FAILED: Found 0 results. The fix isn't working or brand is absent.")

if __name__ == "__main__":
    test_recall_cliff()