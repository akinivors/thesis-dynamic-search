import json
import time
import numpy as np
import os
import pandas as pd
from src.engine import ThesisEngine

class ThesisExperimentV2:
    def __init__(self):
        print("=== INITIALIZING EXPERIMENT V2 (Clean Architecture) ===")
        self.engine = ThesisEngine()
        self.results_summary = []
        self.detailed_logs = []  # Stores full ID lists
        
        # Load Data
        dataset_path = 'data/meta_Electronics.json'
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        self.engine.load_data(dataset_path, limit=None)

    def load_ground_truth(self, filepath="ground_truth.json"):
        with open(filepath, 'r') as f:
            self.ground_truth = json.load(f)
        print(f"[Experiment] Loaded {len(self.ground_truth)} ground truth cases.")

    # ==========================================
    # HELPER: STRICT RECALL CALCULATION
    # ==========================================
    def calculate_recall(self, returned_ids, ground_truth_ids, k_target):
        """
        Strict Recall@K.
        We compare the returned top-K against the TRUE top-K.
        """
        # We only care about the top k_target items from the ground truth
        # (Assuming ground_truth_ids are sorted by relevance/distance)
        relevant_set = set(ground_truth_ids[:k_target])
        
        if not relevant_set:
            return 0.0
            
        matches = len(set(returned_ids).intersection(relevant_set))
        return (matches / k_target) * 100

    # ==========================================
    # SEARCH METHODS (Using Clean Engine Architecture)
    # ==========================================
    # All methods use the engine's clean implementations.
    # We measure end-to-end wall-clock time for fairness.
    # ==========================================
    
    def run_basic_post_filter(self, query_vec, filter_func, filter_desc, k_target=10):
        """
        METHOD 2: Basic Post-Filter (Naive Baseline)
        Uses engine.search_post_filter_basic with k_scan=100
        """
        start = time.perf_counter()
        
        ids, _ = self.engine.search_post_filter_basic(
            query_vec, filter_func, filter_desc, k=k_target, k_scan=100
        )
        
        duration = (time.perf_counter() - start) * 1000
        return ids, duration

    def run_bitmap_hnsw(self, query_vec, filter_func, filter_desc, k_target=10):
        """
        METHOD 1: Bitmap HNSW (Pre-Filter with IDSelector)
        Uses engine.search_bitmap_hnsw - the baseline to beat.
        """
        start = time.perf_counter()
        
        ids, _ = self.engine.search_bitmap_hnsw(
            query_vec, filter_func, filter_desc, k=k_target
        )
        
        duration = (time.perf_counter() - start) * 1000
        return ids, duration

    def run_adaptive_post_filter(self, query_vec, filter_func, filter_desc, k_target=10):
        """
        METHOD 3: Adaptive Post-Filter (Iterative Expansion)
        Uses engine.search_post_filter_adaptive - our innovation.
        """
        start = time.perf_counter()
        
        ids, _ = self.engine.search_post_filter_adaptive(
            query_vec, filter_func, filter_desc, k=k_target
        )
        
        duration = (time.perf_counter() - start) * 1000
        return ids, duration

    # ==========================================
    # MAIN EXECUTION
    # ==========================================
    def run_all(self):
        print("\n=== STARTING CLEAN EXPERIMENTS ===")
        print("Testing 3 methods: Basic Post-Filter, Bitmap HNSW, Adaptive Post-Filter")
        
        # Pre-encode queries to remove ML latency from the equation
        print("Vectorizing queries...")
        query_vectors = self.engine.model.encode([case['query_text'] for case in self.ground_truth])
        
        for i, case in enumerate(self.ground_truth):
            q_vec = query_vectors[i]
            true_ids = case['true_ids']  # Full list (20 items)
            k_target = 10
            
            # --- Setup Filter ---
            f_type = case['filter_type']
            f_val = case['filter_val']
            filter_desc = case['filter_desc']
            
            # Create filter function (same for all methods)
            if f_type == 'price':
                filter_func = lambda idx, v=f_val: self.engine.item_prices[idx] > v
            elif f_type == 'rating':
                filter_func = lambda idx, v=f_val: self.engine.item_ratings[idx] > v
            elif f_type == 'brand':
                filter_func = lambda idx, v=f_val: self.engine.item_brands[idx] == v
            elif f_type == 'category':
                filter_func = lambda idx, v=f_val: self.engine.item_categories[idx] == v
            else:
                print(f"  [SKIP] Unknown filter type: {f_type}")
                continue
            
            # --- METHOD 2: BASIC POST-FILTER (Naive Baseline) ---
            basic_ids, basic_time = self.run_basic_post_filter(q_vec, filter_func, filter_desc, k_target)
            basic_recall = self.calculate_recall(basic_ids, true_ids, k_target)
            
            # --- METHOD 1: BITMAP HNSW (Pre-Filter - Baseline to Beat) ---
            bitmap_ids, bitmap_time = self.run_bitmap_hnsw(q_vec, filter_func, filter_desc, k_target)
            bitmap_recall = self.calculate_recall(bitmap_ids, true_ids, k_target)
            
            # --- METHOD 3: ADAPTIVE POST-FILTER (Our Innovation) ---
            adapt_ids, adapt_time = self.run_adaptive_post_filter(q_vec, filter_func, filter_desc, k_target)
            adapt_recall = self.calculate_recall(adapt_ids, true_ids, k_target)
            
            # --- RECORD RESULTS ---
            
            # 1. Summary (for CSV)
            self.results_summary.append({
                "Selectivity_Zone": self.get_zone(filter_desc),
                "Filter_Desc": filter_desc,
                "Metric_Basic_Lat": basic_time,
                "Metric_Basic_Recall": basic_recall,
                "Metric_Bitmap_Lat": bitmap_time,
                "Metric_Bitmap_Recall": bitmap_recall,
                "Metric_Adaptive_Lat": adapt_time,
                "Metric_Adaptive_Recall": adapt_recall
            })
            
            # 2. Detailed Log (for JSON - Debugging)
            self.detailed_logs.append({
                "query": case['query_text'],
                "filter": filter_desc,
                "true_top_10": true_ids[:10],
                "basic_returned": basic_ids,
                "bitmap_returned": bitmap_ids,
                "adaptive_returned": adapt_ids,
            })
            
            if i % 10 == 0:
                print(f"Processed {i}/{len(self.ground_truth)} queries...")

        self.save_outputs()

    def get_zone(self, desc):
        if "Rare" in desc: return "1. Rare (<1%)"
        if "Medium" in desc: return "2. Medium (1-10%)"
        return "3. Common (>10%)"

    def save_outputs(self):
        # 1. Save Detailed JSON
        with open("thesis_experiment_details.json", "w") as f:
            json.dump(self.detailed_logs, f, indent=2)
        print("\n[Saved] Full returned product lists -> 'thesis_experiment_details.json'")
        
        # 2. Save Clean CSV
        df = pd.DataFrame(self.results_summary)
        
        # Group by Zone for the "Human Eye" table
        summary = df.groupby('Selectivity_Zone').agg({
            'Metric_Basic_Lat': 'mean',
            'Metric_Basic_Recall': 'mean',
            'Metric_Bitmap_Lat': 'mean', 
            'Metric_Bitmap_Recall': 'mean',
            'Metric_Adaptive_Lat': 'mean',
            'Metric_Adaptive_Recall': 'mean'
        }).round(2)
        
        # Rename columns for the final report
        summary.columns = [
            'Basic PostFilter Latency (ms)', 'Basic PostFilter Recall (%)',
            'Bitmap HNSW Latency (ms)', 'Bitmap HNSW Recall (%)',
            'Adaptive PostFilter Latency (ms)', 'Adaptive PostFilter Recall (%)'
        ]
        
        print("\n=== FINAL THESIS DATA (Averaged by Selectivity Zone) ===")
        print(summary.to_string())
        
        summary.to_csv("thesis_summary_table.csv")
        df.to_csv("thesis_raw_results.csv", index=False)
        print("\n[Saved] Summary table -> 'thesis_summary_table.csv'")
        print("[Saved] Raw data -> 'thesis_raw_results.csv'")

if __name__ == "__main__":
    exp = ThesisExperimentV2()
    exp.load_ground_truth()
    exp.run_all()