import argparse
import sys
import os
import json
import pandas as pd
from pathlib import Path

# Fix path to import engine from src/
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

# Local Imports
from .config import Situation5Config, config as global_config
from .experiment import Situation5Experiment
# Import the actual analysis functions (Step 8 completed)
from .analysis import analyze_results, analyze_validation_json

# Import ThesisEngine
try:
    from engine import ThesisEngine
except ImportError:
    print("!! Error: Could not import 'engine'. Check your src/ folder structure.")
    sys.exit(1)

def setup_connections():
    """
    Setup ThesisEngine (NetworkX + FAISS)
    """
    print("   [System] Initializing Thesis Engine...")
    engine = ThesisEngine()
    
    # Point to data file
    data_path = global_config.data_path
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        return None
        
    engine.load_data(data_path, limit=None) # Load full data
    return engine

def run_quick_mode(args):
    """
    Run a quick sanity check (1 query) to ensure pipeline works.
    """
    print("\n" + "=" * 70)
    print("🚀 QUICK MODE: Testing Adaptive-K Implementation")
    print("=" * 70)
    
    engine = setup_connections()
    if not engine: return
    
    # Initialize Experiment
    experiment = Situation5Experiment(engine)
    
    # Generate 1 random query for quick test
    queries = experiment.generate_queries(n_queries=1)
    
    # Run
    experiment.run_experiment(queries=queries)

def run_full_mode(args):
    """
    Run the standard benchmarking suite (Returns CSV).
    This compares all baseline and adaptive strategies across many queries.
    """
    print("\n" + "=" * 70)
    print("📊 FULL MODE: Complete Situation 5 Experiments (CSV Benchmark)")
    print("=" * 70)
    
    engine = setup_connections()
    if not engine: return
    
    # Load default config
    config = Situation5Config()
    
    # Override if args provided
    if args.n_queries:
        config.num_queries = args.n_queries
    
    # Run experiment
    experiment = Situation5Experiment(engine, config)
    results_df = experiment.run_experiment()
    
    # Run Analysis immediately
    if not args.skip_analysis:
        print("\n[Analysis] Generating standard charts...")
        analyze_results(results_df, output_dir=config.results_dir)

def run_validation_mode(args):
    """
    Run the Massive Stress Test / Grand Validation (Returns JSON).
    This focuses on the 4 key strategies for the final thesis audit.
    """
    print("\n" + "=" * 70)
    print("🚀 VALIDATION MODE: Grand Stress Test (JSON Audit)")
    print("=" * 70)

    engine = setup_connections()
    if not engine: return
    
    # Initialize Experiment
    experiment = Situation5Experiment(engine)
    
    # Run the Grand Validation (returns path to JSON file)
    json_path = experiment.run_stress_test(n_queries=args.n_queries)
    
    # Run Validation Analysis immediately on that JSON file
    print("\n[Analysis] Generating Validation Plots...")
    analyze_validation_json(json_path)

def run_analyze_mode(args):
    """
    Analyze existing results from CSV or JSON file without re-running experiments.
    """
    print("\n" + "=" * 70)
    print("📈 ANALYZE MODE: Analyzing Existing Results")
    print("=" * 70)
    
    if not args.results:
        print("❌ Error: --results path required for analyze mode")
        return

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"❌ Error: Results file not found: {results_path}")
        return
    
    print(f"   Loading data from {results_path}...")
    
    # Detect file type and route to appropriate analyzer
    if str(results_path).endswith('.json'):
        print("   Detected JSON format (Validation Run)")
        analyze_validation_json(str(results_path))
    else:
        print("   Detected CSV format (Standard Benchmark)")
        df = pd.read_csv(results_path)
        output_dir = results_path.parent
        analyze_results(df, output_dir=output_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Situation 5: Adaptive-K Vector -> Graph Experiments"
    )
    
    parser.add_argument(
        '--mode',
        choices=['quick', 'full', 'validation', 'analyze'],
        default='quick',
        help='Experiment mode: quick (test), full (csv benchmark), validation (json stress test), analyze (plots only)'
    )
    
    parser.add_argument(
        '--n-queries',
        type=int,
        default=50,
        help='Number of queries per configuration'
    )
    
    parser.add_argument(
        '--results',
        type=str,
        default=None,
        help='Path to results file (CSV or JSON) for analyze mode'
    )
    
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Skip analysis after full experiment'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        run_quick_mode(args)
    elif args.mode == 'full':
        run_full_mode(args)
    elif args.mode == 'validation':
        run_validation_mode(args)
    elif args.mode == 'analyze': 
        run_analyze_mode(args)

if __name__ == '__main__':
    main()