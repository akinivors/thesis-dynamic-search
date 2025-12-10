# Adaptive Graph-Vector Search Optimization

## Abstract
This project implements a **Cost-Based Query Optimizer** for hybrid Graph-Vector databases. Unlike static systems (e.g., TigerVector) that rely on fixed heuristics like pre-filtering, this system dynamically selects an execution plan based on data density and graph topology.

## Key Findings
Our benchmarks on the Amazon Electronics dataset (786k nodes) demonstrate:
1.  **Latency Reduction:** Dynamic switching improves query speed by **4x** on dense attribute queries (e.g., "Sony products") by avoiding expensive pre-filtering.
2.  **Recall Assurance:** The system guarantees **100% recall** for rare items (e.g., "Lensse lens"), where standard vector search fails.
3.  **Graph-First Advantage:** For compatibility queries (e.g., "Find accessories for Hub X"), the graph-first approach is **30x faster** than global vector search.

## Architecture
- **Engine:** Hybrid NetworkX (Graph) + FAISS (Vector)
- **Model:** all-MiniLM-L6-v2 (Sentence Transformers)
- **Optimizer:** Selectivity-based Cost Model with adaptive thresholds.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the experiment you desire: `python experiments/run_(experiment).py`