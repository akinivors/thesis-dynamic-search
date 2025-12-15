import time
import numpy as np
import faiss
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

@dataclass
class SearchResult:
    """Single search result"""
    id: int
    title: str
    score: float  # L2 Distance (Lower is better)
    source_ids: List[int] = field(default_factory=list)  # Which vector anchors led to this

@dataclass
class ExecutionMetrics:
    """Metrics collected during execution"""
    elapsed_ms: float = 0
    vectors_fetched: int = 0
    edges_traversed: int = 0
    unique_targets_found: int = 0
    iterations: int = 1
    k_vector_used: int = 0
    k_final_requested: int = 0
    target_achieved: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionResult:
    """Result of a Vector -> Graph execution"""
    results: List[SearchResult]
    metrics: ExecutionMetrics
    strategy: str

class BaselineVectorToGraph:
    """
    Baseline (Static-K) Vector → Graph implementation.
    Adapts the abstract logic to work with ThesisEngine (FAISS + NetworkX).
    """
    
    def __init__(self, engine):
        """
        Args:
            engine: Your ThesisEngine instance (contains graph + vector_index)
        """
        self.engine = engine
    
    def execute(
        self,
        query_vec: np.ndarray,
        k_final: int,
        edge_type: str = "RELATED" # Unused in NetworkX but kept for interface compatibility
    ) -> ExecutionResult: 
        """
        Execute static-k Vector → Graph
        Strategy: Fetch exactly k_final vectors -> Expand -> Return.
        """
        start_time = time.perf_counter()
        
        # Static: We fetch exactly what the user asked for (Naive approach)
        k_vector = k_final 
        
        metrics = ExecutionMetrics(
            k_final_requested=k_final,
            k_vector_used=k_vector
        )
        
        # Step 1: Vector search
        # FAISS search
        D, I = self.engine.vector_index.search(query_vec.reshape(1, -1), k_vector)
        scores = D[0]
        indices = I[0]
        
        metrics.vectors_fetched = len(indices)
        
        # Step 2: Graph traversal from vector results
        unique_targets: Dict[int, SearchResult] = {}
        
        for i, source_id in enumerate(indices):
            if source_id == -1: continue
            
            source_score = float(scores[i])
            
            # Traverse to targets (Graph Expansion)
            targets = self._traverse(source_id)
            metrics.edges_traversed += len(targets)
            
            for target_id in targets:
                # If target is not in the graph (dangling check), skip
                if not self.engine.graph.has_node(target_id): continue
                
                if target_id not in unique_targets: 
                    # Fetch title for readable results
                    title = self.engine.graph.nodes[target_id].get('title', 'Unknown')
                    unique_targets[target_id] = SearchResult(
                        id=target_id,
                        title=title,
                        score=source_score, # Inherit score from anchor
                        source_ids=[source_id]
                    )
                else:
                    # Update if better score (L2: Min is better)
                    existing = unique_targets[target_id]
                    existing.source_ids.append(source_id)
                    existing.score = min(existing.score, source_score)
        
        # Step 3: Sort and return
        # Sort by Score Ascending (L2 Distance)
        sorted_results = sorted(
            unique_targets.values(),
            key=lambda x: x.score,
            reverse=False 
        )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        metrics.elapsed_ms = elapsed_ms
        metrics.unique_targets_found = len(unique_targets)
        metrics.target_achieved = len(unique_targets) >= k_final
        
        return ExecutionResult(
            results=sorted_results[:k_final],
            metrics=metrics,
            strategy="static_k"
        )
    
    def _traverse(self, source_id: int) -> List[int]:
        """
        Traverse from source vertex using NetworkX
        """
        # In our engine, all edges are "RELATED", so we just get neighbors
        return list(self.engine.graph.neighbors(source_id))

class StaticKWithOverfetch(BaselineVectorToGraph):
    """
    Variant: Static-K with fixed overfetch multiplier.
    This simulates a "smarter" but still static system (e.g. "Always fetch 3x").
    """
    
    def __init__(self, engine, overfetch_multiplier: float = 2.0):
        super().__init__(engine)
        self.overfetch_multiplier = overfetch_multiplier
    
    def execute(
        self,
        query_vec: np.ndarray,
        k_final: int,
        edge_type: str = "RELATED"
    ) -> ExecutionResult: 
        """
        Execute with fixed overfetch
        """
        start_time = time.perf_counter()
        
        # The Static "Fix": Just fetch more vectors blindly
        k_vector = int(k_final * self.overfetch_multiplier)
        
        metrics = ExecutionMetrics(
            k_final_requested=k_final,
            k_vector_used=k_vector
        )
        metrics.extra['overfetch_multiplier'] = self.overfetch_multiplier
        
        # Step 1: Vector search
        D, I = self.engine.vector_index.search(query_vec.reshape(1, -1), k_vector)
        scores = D[0]
        indices = I[0]
        
        metrics.vectors_fetched = len(indices)
        
        # Step 2: Graph traversal
        unique_targets: Dict[int, SearchResult] = {}
        
        for i, source_id in enumerate(indices):
            if source_id == -1: continue
            
            source_score = float(scores[i])
            
            targets = self._traverse(source_id)
            metrics.edges_traversed += len(targets)
            
            for target_id in targets:
                if not self.engine.graph.has_node(target_id): continue

                if target_id not in unique_targets:
                    title = self.engine.graph.nodes[target_id].get('title', 'Unknown')
                    unique_targets[target_id] = SearchResult(
                        id=target_id,
                        title=title,
                        score=source_score,
                        source_ids=[source_id]
                    )
                else:
                    existing = unique_targets[target_id]
                    existing.source_ids.append(source_id)
                    existing.score = min(existing.score, source_score)
        
        # Step 3: Sort and return
        sorted_results = sorted(
            unique_targets.values(),
            key=lambda x: x.score,
            reverse=False 
        )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        metrics.elapsed_ms = elapsed_ms
        metrics.unique_targets_found = len(unique_targets)
        metrics.target_achieved = len(unique_targets) >= k_final
        
        return ExecutionResult(
            results=sorted_results[:k_final],
            metrics=metrics,
            strategy=f"static_k_overfetch_{self.overfetch_multiplier}x"
        )