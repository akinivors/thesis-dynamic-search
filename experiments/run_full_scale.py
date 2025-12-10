import numpy as np
import faiss
from src.engine import ThesisEngine

def run_strict_inspection():
    # 1. Initialize
    engine = ThesisEngine()
    # LOAD EVERYTHING (Warning: Can take time)
    engine.load_data("meta_Electronics.json")

    # 2. Define Diverse Queries
    queries = [
        "noise cancelling headphones",
        "gaming laptop high performance",
        "dslr camera lens"
    ]

    print(f"\n{'='*80}")
    print("FULL-SCALE DATASET VERIFICATION")
    print("Objective: Inspect every fetched item for correctness.")
    print(f"{'='*80}")

    for q_text in queries:
        print(f"\n\n{'#'*60}")
        print(f"QUERY CONTEXT: '{q_text}'")
        print(f"{'#'*60}")
        
        # A. Vectorize
        q_vec = engine.model.encode([q_text])[0].astype('float32')
        faiss.normalize_L2(q_vec.reshape(1, -1))
        
        # B. Find Contextual Brands (The "Smart" Targeting)
        print("   -> Scanning vector space for relevant brands...")
        targets = engine.get_contextual_brands(q_vec)
        
        for tier, (brand, local_count) in targets.items():
            if not brand:
                print(f"   [Skip] Could not find a suitable '{tier}' brand for this query.")
                continue
                
            print(f"\n   >>> TARGET: {tier} Brand = '{brand}' (Local Neighborhood Count: {local_count})")
            
            # --- EXECUTE PRE-FILTER ---
            res_pre, t_pre = engine.search_pre_filter(q_vec, brand, k=10)
            details_pre = engine.get_details(res_pre)
            
            # --- EXECUTE POST-FILTER ---
            res_post, t_post = engine.search_post_filter(q_vec, brand, k=10)
            details_post = engine.get_details(res_post)
            
            # --- STRICT AUDIT LOG ---
            print(f"   [A] PRE-FILTER ({t_pre:.4f}s) | Found: {len(res_pre)}")
            for i, item in enumerate(details_pre):
                print(f"       {i+1}. {item}")
                
            print(f"   [B] POST-FILTER ({t_post:.4f}s) | Found: {len(res_post)}")
            if not details_post:
                print("       (NO RESULTS)")
            for i, item in enumerate(details_post):
                # Verify correctness visually
                marker = " [MATCH]" if item in details_pre else " [DIFF]"
                print(f"       {i+1}. {item}{marker}")

            # --- VERDICT ---
            if len(res_post) < len(res_pre):
                 print(f"   !! VERDICT: POST-FILTER FAILED (Missed {len(res_pre) - len(res_post)} items)")
            elif t_post < t_pre and len(res_post) == len(res_pre):
                 print(f"   !! VERDICT: POST-FILTER WON (Faster & Accurate)")
            else:
                 print(f"   !! VERDICT: TIED or INCONCLUSIVE")

if __name__ == "__main__":
    run_strict_inspection()