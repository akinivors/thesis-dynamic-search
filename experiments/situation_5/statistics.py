import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
import json
import random

@dataclass
class EdgeTypeStatistics:
    """
    Statistics for the graph topology.
    Used to estimate 'Expansion Explosion' risk.
    """
    edge_type: str
    
    # Degree statistics (Fan-out)
    avg_degree: float
    std_degree: float
    min_degree: int
    max_degree: int
    median_degree: float
    p90_degree: float
    p99_degree: float
    
    # Cardinalities
    total_edges: int
    unique_nodes: int
    
    # Derived metrics
    coefficient_of_variation: float  # std/mean (High = Some hubs are massive)
    overlap_factor: float  # High = Neighbors cluster together
    
    def to_dict(self) -> dict:
        return self.__dict__

class StatisticsCollector:
    """
    Collects and caches graph statistics for Adaptive-K estimation.
    Adapted for ThesisEngine (NetworkX).
    """
    
    def __init__(self, engine):
        """
        Args:
            engine: Instance of ThesisEngine containing the loaded graph
        """
        self.engine = engine
        self.cache: Dict[str, EdgeTypeStatistics] = {}
    
    def get_statistics(self, edge_type: str = "RELATED", force_refresh: bool = False) -> EdgeTypeStatistics:
        """
        Get statistics for the graph structure.
        """
        if edge_type not in self.cache or force_refresh:
            self.cache[edge_type] = self._compute_statistics(edge_type)
        return self.cache[edge_type]
    
    def get_vertex_degree(self, vertex_id: int) -> int:
        """Get degree of a specific vertex"""
        return self.engine.graph.degree[vertex_id]
    
    def _compute_statistics(self, edge_type: str) -> EdgeTypeStatistics:
        """
        Compute statistics directly from NetworkX graph
        """
        print(f"   [Statistics] Computing topology metrics for '{edge_type}'...")
        
        # 1. Get all degrees
        # (list of degrees for every node)
        degrees = [d for n, d in self.engine.graph.degree()]
        
        if not degrees:
            return EdgeTypeStatistics(edge_type, 0,0,0,0,0,0,0,0,0,0,0)

        # 2. Calculate Distribution Metrics
        avg_deg = np.mean(degrees)
        std_deg = np.std(degrees)
        
        # 3. Calculate Percentiles
        p50, p90, p99 = np.percentile(degrees, [50, 90, 99])
        
        # 4. Calculate CV (Coefficient of Variation)
        # CV > 1 means "Heavy Tailed" (Super Hubs exist)
        cv = std_deg / avg_deg if avg_deg > 0 else 0
        
        # 5. Calculate Overlap
        overlap = self._estimate_overlap_factor()
        
        stats = EdgeTypeStatistics(
            edge_type=edge_type,
            avg_degree=float(avg_deg),
            std_degree=float(std_deg),
            min_degree=int(np.min(degrees)),
            max_degree=int(np.max(degrees)),
            median_degree=float(p50),
            p90_degree=float(p90),
            p99_degree=float(p99),
            total_edges=self.engine.graph.number_of_edges(),
            unique_nodes=self.engine.graph.number_of_nodes(),
            coefficient_of_variation=float(cv),
            overlap_factor=overlap
        )
        
        print(f"   [Statistics] Graph is {'Heavy-Tailed' if cv > 1 else 'Uniform'}. P99 Degree: {stats.p99_degree}")
        return stats
    
    def _estimate_overlap_factor(self, sample_size: int = 1000) -> float:
        """
        Estimates Jaccard Similarity between random node pairs.
        High overlap = "Small World" (Everything connects to everything).
        Low overlap = "Tree Like" (Distinct branches).
        """
        nodes = list(self.engine.graph.nodes())
        if len(nodes) < 2: return 0.0
        
        # Pick random sample
        sample_nodes = random.sample(nodes, min(sample_size, len(nodes)))
        
        total_jaccard = 0
        pair_count = 0
        
        # Compare neighbors of sampled nodes
        for i in range(len(sample_nodes)):
            # Check 5 random partners for each node
            partners = random.sample(nodes, min(5, len(nodes)))
            
            for partner in partners:
                if sample_nodes[i] == partner: continue
                
                # Get neighbors
                set_i = set(self.engine.graph.neighbors(sample_nodes[i]))
                set_j = set(self.engine.graph.neighbors(partner))
                
                if len(set_i) > 0 and len(set_j) > 0:
                    intersection = len(set_i & set_j)
                    union = len(set_i | set_j)
                    jaccard = intersection / union
                    total_jaccard += jaccard
                    pair_count += 1
        
        if pair_count == 0: return 0.0
        return total_jaccard / pair_count