import json
import time
import numpy as np
import os
import pandas as pd
from src.engine import ThesisEngine

class ThesisExperiment:
    def __init__(self):
        self.engine = ThesisEngine()
        self.results = []
        
        # Load Data
        dataset_path = 'data/meta_Electronics.json'
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        self.engine.load_data(dataset_path, limit=None)

    def load_ground_truth(self, filepath="ground_truth.json"):
        with open(filepath, 'r') as f:
            self.ground_truth = json.load(f)
        print(f"\n[Experiment] Loaded {len(self.ground_truth)} ground truth cases.")

    # ==========================================
    # COMPETITOR 1: NAIVE POST-FILTER
    # ==========================================
    def run_naive_post_filter(self, query_vec, filter_func, k_target=10):
        start_time = time.perf_counter()
        
        # Fixed budget (standard industry default for post-filtering)
        k_scan = 100 
        D, I = self.engine.hnsw_index.search(query_vec.reshape(1, -1), k_scan)
        
        found_ids = []
        for idx in I[0]:
            if idx != -1 and filter_func(idx):
                found_ids.append(int(idx))
                if len(found_ids) >= k_target:
                    break
                    
        total_time = (time.perf_counter() - start_time) * 1000
        return found_ids, total_time

    # ==========================================
    # COMPETITOR 2: BITMAP PRE-FILTER (The Standard)
    # ==========================================
    def run_bitmap_pre_filter(self, query_vec, filter_type, val, k_target=10):
        """
        Calls 'search_bitmap_pre_filter' (Global HNSW + IDSelector).
        PURE HNSW TEST: No flat search fallback.
        """
        # FIX: Removed the extra variable unpacking. 
        # The engine method returns (ids, latency).
        ids, latency = self.engine.search_bitmap_pre_filter(
            query_vec, filter_type, val, k=k_target
        )
        return ids, latency * 1000 

    # ==========================================
    # COMPETITOR 3: ADAPTIVE POST-FILTER (The Hero)
    # ==========================================
    def run_adaptive_post_filter(self, query_vec, filter_func, filter_desc, k_target=10):
        """
        Calls 'search_post_filter_adaptive'.
        PURE HNSW TEST: Iterative graph expansion.
        """
        ids, latency = self.engine.search_post_filter_adaptive(
            query_vec, filter_func, filter_desc, k=k_target
        )
        return ids, latency * 1000 

    # ==========================================
    # MAIN LOOP
    # ==========================================
    def run_all(self):
        print("\n=== STARTING THESIS EXPERIMENTS ===")
        print("Protocol: PURE HNSW (No Flat Search Fallbacks allowed)")
        print(f"Comparing 3 Models on {len(self.ground_truth)} queries...")
        
        query_vectors = self.engine.model.encode([case['query_text'] for case in self.ground_truth])
        
        for i, case in enumerate(self.ground_truth):
            q_vec = query_vectors[i]
            true_ids = set(case['true_ids'])
            k_target = 10
            
            # Reconstruct Filter Function
            f_type = case['filter_type']
            f_val = case['filter_val']
            
            if f_type == 'price':
                filter_func = lambda idx, v=f_val: self.engine.item_prices[idx] > v
            elif f_type == 'rating':
                filter_func = lambda idx, v=f_val: self.engine.item_ratings[idx] > v
            elif f_type == 'brand':
                filter_func = lambda idx, v=f_val: self.engine.item_brands[idx] == v
            elif f_type == 'category':
                filter_func = lambda idx, v=f_val: self.engine.item_categories[idx] == v
            
            # --- RUN MODELS ---
            
            # 1. Naive
            naive_ids, naive_time = self.run_naive_post_filter(q_vec, filter_func, k_target)
            naive_recall = len(set(naive_ids).intersection(true_ids)) / len(true_ids) * 100 if true_ids else 0
            
            # 2. Bitmap Pre-Filter
            bitmap_ids, bitmap_time = self.run_bitmap_pre_filter(q_vec, f_type, f_val, k_target)
            bitmap_recall = len(set(bitmap_ids).intersection(true_ids)) / len(true_ids) * 100 if true_ids else 0
            
            # 3. Adaptive
            adaptive_ids, adaptive_time = self.run_adaptive_post_filter(q_vec, filter_func, case['filter_desc'], k_target)
            adaptive_recall = len(set(adaptive_ids).intersection(true_ids)) / len(true_ids) * 100 if true_ids else 0
            
            # Log Result
            self.results.append({
                "Selectivity_Desc": case['filter_desc'],
                "Filter_Type": f_type,
                "Latency_Naive": naive_time,
                "Recall_Naive": naive_recall,
                "Latency_Bitmap": bitmap_time,
                "Recall_Bitmap": bitmap_recall,
                "Latency_Adaptive": adaptive_time,
                "Recall_Adaptive": adaptive_recall
            })
            
            if i % 10 == 0:
                print(f"Processed {i}/{len(self.ground_truth)}...")

        self.save_summary()

    def save_summary(self):
        df = pd.DataFrame(self.results)
        
        # Categorize Selectivity for Analysis
        def get_bucket(desc):
            if "Rare" in desc: return "1. Rare (<1%)"
            if "Medium" in desc: return "2. Medium (1-10%)"
            return "3. Common (>10%)"
            
        df['Bucket'] = df['Selectivity_Desc'].apply(get_bucket)
        
        print("\n=== FINAL RESULTS SUMMARY ===")
        summary = df.groupby('Bucket').agg({
            'Latency_Naive': 'mean',
            'Recall_Naive': 'mean',
            'Latency_Bitmap': 'mean', 
            'Recall_Bitmap': 'mean',
            'Latency_Adaptive': 'mean',
            'Recall_Adaptive': 'mean'
        }).sort_index()
        
        print(summary.to_string())
        
        df.to_csv("final_thesis_results.csv", index=False)
        print("\nFull results saved to 'final_thesis_results.csv'")

if __name__ == "__main__":
    exp = ThesisExperiment()
    exp.load_ground_truth()
    exp.run_all()