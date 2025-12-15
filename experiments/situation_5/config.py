from dataclasses import dataclass, field
from typing import List
import json
import os

@dataclass
class Situation5Config:
    """
    Configuration for Situation 5 (Vector -> Graph Expansion)
    Adapted for ThesisEngine (Embedded FAISS + NetworkX)
    """
    # Paths
    # We point dynamically to the data folder relative to this file
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    data_path: str = os.path.join(base_dir, '..', '..', 'data', 'meta_Books.jsonl')
    results_dir: str = os.path.join(base_dir, '..', '..', 'data', 'results', 'situation_5')
    
    # Vector Settings
    model_name: str = 'all-MiniLM-L6-v2'
    vector_dimension: int = 384  # Correct dim for MiniLM
    
    # Experiment Rigor
    # We will test our Adaptive method against these fixed K baselines
    k_fixed_baselines: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    
    # How many random "Hubs" to test against
    num_queries: int = 50 
    random_seed: int = 42
    
    # The "Elbow" parameters we tuned
    elbow_sensitivity: str = "dynamic" # or "strict"
    
    def save(self):
        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=str)

# Create global instance
config = Situation5Config()