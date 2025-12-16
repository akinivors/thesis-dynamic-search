import time
import numpy as np
import faiss
from src.engine import ThesisEngine

def profile_system():
    print("="*60)
    print("⚡️ SYSTEM CALIBRATION & PROFILING ⚡️")
    print("="*60)
    
    # Initialize Engine (Loads data/vectors)
    engine = ThesisEngine()
    engine.load_data("data/meta_Electronics.json")
    
    # Generate a dummy query
    dim = engine.d
    q_vec = np.random.randn(dim).astype('float32')
    faiss.normalize_L2(q_vec.reshape(1, -1))
    
    # --- METRIC 1: Cost of Vector Distance Calculation (Pre-Filter) ---
    # Measure time to compute exact distance for N vectors (Brute Force)
    print("\n[1] Profiling Vector Math (Pre-Filter Cost)...")
    
    N_samples = [100, 1000, 10000]
    times = []
    
    # Extract some real vectors to calculate against
    real_vecs = engine.vectors[:10000]
    
    for N in N_samples:
        subset = real_vecs[:N]
        
        start = time.perf_counter()
        # Simulate what Pre-Filter does: Dot product + Sort
        scores = np.dot(subset, q_vec.T)
        _ = np.argsort(scores, axis=0)
        duration = time.perf_counter() - start
        
        times.append(duration)
        print(f"    N={N}: {duration*1000:.4f} ms")

    # Calculate Time Per Item (Slope)
    # T_total = Intercept + N * T_per_item
    # We'll just average the per-item cost for simplicity
    t_per_item = np.mean([t/n for t, n in zip(times, N_samples)])
    print(f"    >> T_distance_calc ≈ {t_per_item*1000:.6f} ms/item")
    
    # --- METRIC 2: Cost of Index Search (Post-Filter) ---
    print("\n[2] Profiling Index Search (Post-Filter Cost)...")
    
    k_attempts = [500, 1000, 2000] # Typical oversampling sizes
    index_times = []
    
    for k in k_attempts:
        start = time.perf_counter()
        _, _ = engine.vector_index.search(q_vec.reshape(1, -1), k)
        duration = time.perf_counter() - start
        
        index_times.append(duration)
        print(f"    k_search={k}: {duration*1000:.4f} ms")
        
    avg_index_time = np.mean(index_times)
    print(f"    >> T_index_search (avg) ≈ {avg_index_time*1000:.4f} ms")

    # --- THE CROSSOVER POINT ---
    # C_pre = N * t_per_item
    # C_post = T_index_search (simplified, ignoring attr check which is tiny)
    # Equating: N * t_per_item = T_index_search
    # N_threshold = T_index_search / t_per_item
    
    threshold = avg_index_time / t_per_item
    
    print("\n" + "="*60)
    print(f"🔬 CALCULATED OPTIMAL THRESHOLD: {int(threshold)} items")
    print("="*60)
    print("Update your src/engine.py with this value, or implement the dynamic formula!")

if __name__ == "__main__":
    profile_system()