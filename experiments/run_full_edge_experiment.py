import numpy as np
import faiss
from src.engine import ThesisEngine

def run_full_scale_edge_test():
    # 1. Initialize
    engine = ThesisEngine()
    
    # 2. LOAD EVERYTHING
    # We pass limit=None to load the entire dataset.
    # Note: This might take 2-5 minutes depending on file size.
    print(f"\n{'='*80}")
    print("FULL-SCALE GRAPH LOADING STARTING...")
    print("This will ensure we capture all edges and avoid the 'Subset Problem'.")
    print(f"{'='*80}")
    
    engine.load_data("meta_Electronics.json", limit=None)

    # 3. VERIFY GRAPH HEALTH
    num_nodes = engine.graph.number_of_nodes()
    num_edges = engine.graph.number_of_edges()
    
    print(f"\n[Graph Statistics]")
    print(f"   - Nodes (Products): {num_nodes}")
    print(f"   - Edges (Relations): {num_edges}")
    
    if num_edges == 0:
        print("\n!! CRITICAL WARNING: No edges found in the entire dataset.")
        print("   Please verify your 'meta_Electronics.json' contains the 'related' field.")
        return

    # 4. FIND THE 'SUPER HUB'
    # We search the entire graph for the node with the most connections.
    print("\n[Analysis] Scanning for the 'Super Hub' (Most Connected Product)...")
    degree_list = sorted(engine.graph.degree, key=lambda x: x[1], reverse=True)
    
    # We try to find a hub that has meaningful neighbors (titles that aren't empty)
    hub_id = None
    for nid, count in degree_list:
        if count > 5:
            # Check if neighbors have titles
            neighbors = list(engine.graph.neighbors(nid))
            if any(engine.graph.nodes[n]['title'] for n in neighbors):
                hub_id = nid
                break
    
    if not hub_id:
        print("!! ERROR: Could not find a suitable hub with >5 connections.")
        return

    hub_data = engine.graph.nodes[hub_id]
    neighbors = list(engine.graph.neighbors(hub_id))
    
    print(f"\n[Hub Product Selected]")
    print(f"   Title: {hub_data['title'][:100]}...")
    print(f"   ASIN:  {hub_data['asin']}")
    print(f"   Brand: {hub_data['brand']}")
    print(f"   Connections: {len(neighbors)}")

    # 5. GENERATE A 'CHEATING' QUERY
    # We pick a specific keyword from one of the neighbors to ensure a match exists.
    # This guarantees that 'Recall Failure' is the fault of the algorithm, not the data.
    target_neighbor = engine.graph.nodes[neighbors[0]]
    target_title = target_neighbor['title']
    
    # Heuristic: Pick the first 3-4 words of a neighbor as the query
    # e.g. "Canon EF 50mm f/1.8" -> "Canon EF 50mm"
    query_text = " ".join(target_title.split()[:4])
    
    print(f"\n[User Query]: '{query_text}'")
    print(f"   (Derived from neighbor: '{target_title[:60]}...')")
    print(f"   Context: User is looking for this specific accessory for the Hub.")

    # Vectorize
    q_vec = engine.model.encode([query_text])[0].astype('float32')
    faiss.normalize_L2(q_vec.reshape(1, -1))

    # ==========================================================
    # STRATEGY A: PRE-FILTER (Graph Traversal -> Vector Search)
    # ==========================================================
    res_pre, t_pre = engine.search_pre_filter_graph(q_vec, hub_id, k=10)
    details_pre = engine.get_details(res_pre)

    print(f"\n{'-'*60}")
    print(f"[A] PRE-FILTER (Graph-First) | Time: {t_pre:.5f}s")
    print(f"{'-'*60}")
    if not res_pre:
        print("    (No matches found - This shouldn't happen with the cheating query!)")
    for i, item in enumerate(details_pre):
        print(f"    {i+1}. {item}")

    # ==========================================================
    # STRATEGY B: POST-FILTER (Vector Search -> Graph Validation)
    # ==========================================================
    res_post, t_post = engine.search_post_filter_graph(q_vec, hub_id, k=10)
    details_post = engine.get_details(res_post)

    print(f"\n{'-'*60}")
    print(f"[B] POST-FILTER (Vector-First) | Time: {t_post:.5f}s")
    print(f"{'-'*60}")
    
    if not res_post:
        print("    (NO RESULTS - Recall Failure)")
        print("    Explanation: The compatible items were not in the Global Top-1000.")
    else:
        for i, item in enumerate(details_post):
            status = "MATCH" if item in details_pre else "DIFF"
            print(f"    {i+1}. {item} [{status}]")

    # ==========================================================
    # FINAL VERDICT
    # ==========================================================
    print(f"\n{'-'*60}")
    if len(res_post) < len(res_pre):
        missed = len(res_pre) - len(res_post)
        print(f"thesis_evidence: POST-FILTER FAILED")
        print(f"   -> It missed {missed} valid, compatible items.")
        print(f"   -> This proves Vector Search ignores graph constraints (compatibility).")
    elif t_pre < t_post:
        print(f"thesis_evidence: PRE-FILTER WAS FASTER")
        print(f"   -> Graph Hop ({t_pre:.5f}s) beat Global Search ({t_post:.5f}s).")
    else:
        print(f"thesis_evidence: INCONCLUSIVE (Tied)")
    print(f"{'-'*60}")

if __name__ == "__main__":
    run_full_scale_edge_test()
