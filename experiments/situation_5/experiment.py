import time
import json
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Import local configuration and modules
from .config import Situation5Config, config as global_config
from .statistics import StatisticsCollector
from .baseline import BaselineVectorToGraph, StaticKWithOverfetch
from .adaptive import AdaptiveKVectorToGraph
from .models import get_all_models, EnsembleModel, DistributionAwareModel

@dataclass
class ExperimentQuery:
    """Single experiment query definition"""
    query_id: str
    query_vec: np.ndarray
    edge_type: str
    k_final: int

@dataclass
class ExperimentRun:
    """Detailed result of a single run"""
    query_id: str
    strategy: str
    k_final: int
    target_achieved: bool
    results_found: int
    elapsed_ms: float
    vectors_fetched: int
    iterations: int
    vector_efficiency: float
    # Detailed logs for JSON export
    extra: Dict[str, Any]

class Situation5Experiment:
    """
    Main experiment runner for Situation 5.
    Supports both 'Full Benchmark' (CSV) and 'Grand Validation' (JSON) modes.
    """
    
    def __init__(
        self,
        engine,
        config: Optional[Situation5Config] = None
    ):
        self.engine = engine
        self.config = config or global_config
        
        # Initialize components
        self.stats_collector = StatisticsCollector(engine)
        
        # Initialize standard strategies for the full benchmark
        self.strategies = self._init_strategies()
        
        # Results storage
        self.results: List[ExperimentRun] = []
        
        # Create results directory
        os.makedirs(self.config.results_dir, exist_ok=True)
    
    def _init_strategies(self) -> Dict: 
        """
        Initialize all strategies for general comparison
        """
        strategies = {}
        
        # Baseline strategies (Naive)
        strategies['static_k'] = BaselineVectorToGraph(self.engine)
        strategies['static_2x'] = StaticKWithOverfetch(self.engine, 2.0)
        
        # Adaptive strategies (Smart)
        models = get_all_models()
        for model_name, model in models.items():
            # Single-shot adaptive (Fast)
            strategies[f'adaptive_{model_name}_single'] = AdaptiveKVectorToGraph(
                self.engine,
                self.stats_collector,
                k_model=model
            )
        
        # Iterative adaptive (Safe)
        strategies['adaptive_ensemble_iterative'] = AdaptiveKVectorToGraph(
            self.engine,
            self.stats_collector,
            k_model=models['ensemble']
        )
        
        return strategies

    def generate_queries(self, n_queries: int = None) -> List[ExperimentQuery]:
        """
        Generate N random queries for testing.
        """
        n_queries = n_queries or self.config.num_queries
        queries = []
        np.random.seed(self.config.random_seed)
        edge_type = "RELATED"
        
        # We test across different K requirements (5, 10, 20)
        k_values = [5, 10, 20] 
        
        print(f"   [Generator] Creating {n_queries} unique query vectors...")
        
        # Ensure distinct IDs across K values
        global_id = 0
        
        for k_final in k_values:
            # Distribute N queries across K values
            count_for_k = n_queries // len(k_values)
            # Add remainder to first batch to ensure total count is correct
            if k_final == k_values[0]:
                count_for_k += n_queries % len(k_values)
                
            for _ in range(count_for_k):
                # Generate random query vector
                q_vec = np.random.randn(self.config.vector_dimension).astype('float32')
                q_vec /= np.linalg.norm(q_vec)  # Normalize
                
                queries.append(ExperimentQuery(
                    query_id=f"q_{global_id}",
                    query_vec=q_vec,
                    edge_type=edge_type,
                    k_final=k_final
                ))
                global_id += 1
        
        return queries

    def run_stress_test(self, n_queries: int = 100) -> str:
        """
        The Grand Validation: Test 4 specific strategies on N queries.
        Returns the path to the saved JSON log.
        """
        print("=" * 70)
        print(f"🚀 GRAND VALIDATION: Running {n_queries} Queries")
        print("=" * 70)

        # 1. Define the 4 Strategies to Test (Specific Configurations)
        # 1. Define Strategies
        strategies = {
            'static_k': BaselineVectorToGraph(self.engine),
            
            # The Old "Upward Only" (Batch) approach 
            # (We keep this implicitly if we disabled early exit, but for now 
            # let's assume the new class replaces the old logic. 
            # To prove the gain, we compare against Static-2x and Static-K)
            'static_2x': StaticKWithOverfetch(self.engine, 2.0),
            
            # This is now the ENHANCED Bidirectional Engine
            'adaptive_bidirectional': AdaptiveKVectorToGraph(
                self.engine, self.stats_collector, 
                k_model=EnsembleModel(), 
                
            ),
            
            # Fast mode
            'adaptive_single': AdaptiveKVectorToGraph(
                self.engine, self.stats_collector, 
                k_model=DistributionAwareModel(), 
                
            )
        }

        # 2. Generate Data (Using K=10 specifically for fair comparison)
        queries = []
        np.random.seed(self.config.random_seed)
        for i in range(n_queries):
            q_vec = np.random.randn(self.config.vector_dimension).astype('float32')
            q_vec /= np.linalg.norm(q_vec)
            queries.append(ExperimentQuery(
                query_id=f"q_{i}", query_vec=q_vec, edge_type="RELATED", k_final=10
            ))
            
        self.results = []
        
        # 3. Execution Loop
        for query in tqdm(queries, desc="Benchmarking"):
            for name, strategy in strategies.items():
                try:
                    # Execute
                    if 'adaptive' in name:
                        # Adaptive strategies allow configuration
                        use_iter = 'iterative' in name
                        res = strategy.execute(
                            query.query_vec, query.k_final, 
                            use_iterative=use_iter
                        )
                    else:
                        # Static strategies just take the vector and K
                        res = strategy.execute(query.query_vec, query.k_final)
                    
                    # Calculate Efficiency (Targets found / Vectors fetched)
                    eff = 0.0
                    if res.metrics.vectors_fetched > 0:
                        eff = query.k_final / res.metrics.vectors_fetched
                    
                    # Log Result
                    run_data = ExperimentRun(
                        query_id=query.query_id,
                        strategy=name,
                        k_final=query.k_final,
                        target_achieved=res.metrics.target_achieved,
                        results_found=res.metrics.unique_targets_found,
                        elapsed_ms=res.metrics.elapsed_ms,
                        vectors_fetched=res.metrics.vectors_fetched,
                        iterations=res.metrics.iterations,
                        vector_efficiency=eff,
                        extra=res.metrics.extra
                    )
                    self.results.append(run_data)

                except Exception as e:
                    print(f"!! Error in {name} on {query.query_id}: {e}")

        # 4. Save Detailed JSON (The "Solid Data Output")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(self.config.results_dir, f"validation_log_{timestamp}.json")
        
        # Convert dataclass objects to simple dictionaries for JSON serialization
        export_data = [asdict(r) for r in self.results]
        
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"\n✅ Detailed Validation Log saved to: {json_path}")
        return json_path

    def run_experiment(
        self,
        queries: Optional[List[ExperimentQuery]] = None,
        strategies_to_test: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Run the full standard experiment (returns DataFrame/CSV).
        """
        queries = queries or self.generate_queries()
        strategies_to_test = strategies_to_test or list(self.strategies.keys())
        
        print("=" * 70)
        print("SITUATION 5 EXPERIMENT: Adaptive-K Vector -> Graph")
        print("=" * 70)
        print(f"Queries: {len(queries)}")
        print(f"Strategies: {len(strategies_to_test)}")
        print(f"Total runs: {len(queries) * len(strategies_to_test)}")
        print("=" * 70)
        
        # 1. Stats
        print("\n[1/3] Collecting graph statistics...")
        stats = self.stats_collector.get_statistics("RELATED")
        print(f"   Avg Degree: {stats.avg_degree:.2f}, CV: {stats.coefficient_of_variation:.2f}")
        
        # 2. Run
        print("\n[2/3] Running experiments...")
        self.results = []
        
        for query in tqdm(queries, desc="Queries"):
            for strategy_name in strategies_to_test:
                try:
                    result = self._run_single(query, strategy_name)
                    self.results.append(result)
                except Exception as e:
                    print(f"Error in {strategy_name}: {e}")
                    continue
        
        # 3. Process
        print("\n[3/3] Processing results...")
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.config.results_dir, f"results_{timestamp}.csv")
        df.to_csv(results_path, index=False)
        print(f"\n✓ Results saved to {results_path}")
        
        self._print_summary(df)
        return df
    
    def _run_single(self, query: ExperimentQuery, strategy_name: str) -> ExperimentRun:
        strategy = self.strategies[strategy_name]
        use_iterative = 'iterative' in strategy_name
        
        if isinstance(strategy, AdaptiveKVectorToGraph):
            result = strategy.execute(
                query.query_vec,
                query.k_final, 
                use_iterative=use_iterative
            )
        else:
            result = strategy.execute(
                query.query_vec,
                query.k_final
            )
        
        # Calculate efficiencies
        vec_eff = 0.0
        if result.metrics.vectors_fetched > 0:
            vec_eff = query.k_final / result.metrics.vectors_fetched
            
        return ExperimentRun(
            query_id=query.query_id,
            strategy=strategy_name,
            k_final=query.k_final,
            target_achieved=result.metrics.target_achieved,
            results_found=result.metrics.unique_targets_found,
            elapsed_ms=result.metrics.elapsed_ms,
            vectors_fetched=result.metrics.vectors_fetched,
            iterations=result.metrics.iterations,
            vector_efficiency=vec_eff,
            extra=result.metrics.extra
        )
    
    def _print_summary(self, df: pd.DataFrame):
        print("\n" + "=" * 70)
        print("EXPERIMENT SUMMARY")
        print("=" * 70)
        
        # Success Rate
        print("\n📊 Target Achievement Rate (Did we find K items?):")
        achievement = df.groupby('strategy')['target_achieved'].mean().sort_values(ascending=False)
        for strategy, rate in achievement.items():
            print(f"  {strategy:40s}: {rate*100:6.1f}%")
        
        # Latency
        print("\n⏱️  Average Latency (ms):")
        latency = df.groupby('strategy')['elapsed_ms'].mean().sort_values()
        for strategy, ms in latency.items():
            print(f"  {strategy:40s}: {ms:8.2f} ms")
        
        # Efficiency
        print("\n🎯 Vector Efficiency (Higher is Better - Less Waste):")
        efficiency = df.groupby('strategy')['vector_efficiency'].mean().sort_values(ascending=False)
        for strategy, eff in efficiency.items():
            print(f"  {strategy:40s}: {eff:6.2f}")