import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from pathlib import Path
from typing import Optional

# =========================================================
# PART A: CSV Benchmark Analysis (The "Old" Detailed Code)
# =========================================================

def analyze_results(df: pd.DataFrame, output_dir: Optional[str] = None):
    """
    Comprehensive analysis of experiment results (CSV Format).
    Generates tables, specific breakdowns, and thesis-quality charts.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'results', 'situation_5')
    
    # Ensure it's a Path object and exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("📊 DETAILED THESIS ANALYSIS (BENCHMARK)")
    print("=" * 70)
    
    # 1. Overall comparison
    _analyze_overall(df)
    
    # 2. By k_final (User Constraint)
    _analyze_by_k_final(df)
    
    # 3. By Edge Type (Topology)
    _analyze_by_edge_type(df)
    
    # 4. Model comparison (Math)
    _analyze_models(df)
    
    # 5. Generate plots (Visuals)
    _generate_plots(df, output_dir)
    
    print(f"\n✓ Analysis complete. Charts saved to {output_dir}")

def _analyze_overall(df: pd.DataFrame):
    print("\n" + "-" * 50)
    print("1. OVERALL STRATEGY COMPARISON")
    print("-" * 50)
    
    baseline_strategies = [s for s in df['strategy'].unique() if 'static' in s]
    adaptive_strategies = [s for s in df['strategy'].unique() if 'adaptive' in s]
    
    print(f"Comparing {len(baseline_strategies)} Baseline vs {len(adaptive_strategies)} Adaptive strategies.")
    print("\n" + f"{'Strategy':<40} {'Target%':>10} {'Time(ms)':>10} {'Vectors':>10} {'Efficiency':>10}")
    print("-" * 80)
    
    strategy_stats = df.groupby('strategy').agg({
        'target_achieved': 'mean',
        'elapsed_ms': 'mean',
        'vectors_fetched': 'mean',
        'vector_efficiency': 'mean'
    }).sort_values('target_achieved', ascending=False)

    for strategy, row in strategy_stats.iterrows():
        print(f"{strategy:<40} "
              f"{row['target_achieved']*100:>9.1f}% "
              f"{row['elapsed_ms']:>10.2f} "
              f"{row['vectors_fetched']:>10.1f} "
              f"{row['vector_efficiency']:>10.3f}")

def _analyze_by_k_final(df: pd.DataFrame):
    print("\n" + "-" * 50)
    print("2. ANALYSIS BY QUERY DIFFICULTY (K_FINAL)")
    print("-" * 50)
    
    for k_final in sorted(df['k_final'].unique()):
        subset = df[df['k_final'] == k_final]
        print(f"\n📌 User Requested K = {k_final}")
        
        baseline = subset[subset['strategy'].str.contains('static')]
        adaptive = subset[subset['strategy'].str.contains('adaptive')]
        
        if len(baseline) > 0:
            best_base = baseline.loc[baseline['elapsed_ms'].idxmin()]
            print(f"   Best Static:   {best_base['strategy']:<35} ({best_base['elapsed_ms']:.2f}ms, {best_base['target_achieved']*100:.0f}% Success)")
            
        if len(adaptive) > 0:
            best_adap_speed = adaptive.loc[adaptive['elapsed_ms'].idxmin()]
            best_adap_reli = adaptive.loc[adaptive['target_achieved'].idxmax()]
            
            print(f"   Fastest Adaptive: {best_adap_speed['strategy']:<32} ({best_adap_speed['elapsed_ms']:.2f}ms, {best_adap_speed['target_achieved']*100:.0f}% Success)")

def _analyze_by_edge_type(df: pd.DataFrame):
    print("\n" + "-" * 50)
    print("3. ANALYSIS BY EDGE TYPE")
    print("-" * 50)
    
    if 'edge_type' not in df.columns:
        print("   (No edge type data available)")
        return

    for edge_type in df['edge_type'].unique():
        subset = df[df['edge_type'] == edge_type]
        print(f"\n📌 Edge Type: {edge_type} (Count: {len(subset)})")
        top3 = subset.groupby('strategy')['elapsed_ms'].mean().nsmallest(3)
        for strat, ms in top3.items():
            print(f"   Top Performer: {strat:<30} {ms:.2f}ms")

def _analyze_models(df: pd.DataFrame):
    print("\n" + "-" * 50)
    print("4. MATH MODEL COMPARISON")
    print("-" * 50)
    
    adaptive = df[df['strategy'].str.contains('adaptive')].copy()
    if len(adaptive) == 0: return

    if 'model_name' not in adaptive.columns or adaptive['model_name'].isnull().all():
        adaptive['model_name'] = adaptive['strategy'].apply(
            lambda s: s.replace('adaptive_', '').replace('_single', '').replace('_iterative', '')
        )
    
    summary = adaptive.groupby('model_name').agg({
        'target_achieved': 'mean',
        'elapsed_ms': 'mean',
        'vector_efficiency': 'mean'
    }).sort_values('elapsed_ms')
    print(summary)

def _generate_plots(df: pd.DataFrame, output_dir: Path):
    print("\n[Graphics] Generating plots...")
    sns.set_style("whitegrid")
    
    # 1. LATENCY
    plt.figure(figsize=(12, 6))
    chart = sns.barplot(data=df, x='strategy', y='elapsed_ms', hue='k_final', errorbar=None)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Average Latency by Strategy (Lower is Better)')
    plt.ylabel('Latency (ms)')
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_comparison.png')
    plt.close()
    
    # 2. SUCCESS RATE
    plt.figure(figsize=(12, 6))
    chart = sns.barplot(data=df, x='strategy', y='target_achieved', hue='k_final', errorbar=None)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Reliability: Target Success Rate (Higher is Better)')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rate.png')
    plt.close()
    
    # 3. VECTOR EFFICIENCY
    plt.figure(figsize=(12, 6))
    chart = sns.barplot(data=df, x='strategy', y='vector_efficiency', errorbar=None)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Vector Efficiency: Results per Fetch (Higher is Better)')
    plt.tight_layout()
    plt.savefig(output_dir / 'vector_efficiency.png')
    plt.close()

    # 4. PARETO FRONTIER
    summary = df.groupby('strategy').agg({
        'elapsed_ms': 'mean',
        'target_achieved': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=summary, x='elapsed_ms', y='target_achieved', hue='strategy', s=200, style='strategy')
    for i, row in summary.iterrows():
        short_name = row['strategy'].replace('adaptive_', '').replace('static_', '').replace('_single', '').replace('_iterative', '')
        plt.text(row['elapsed_ms']+0.2, row['target_achieved']+0.005, short_name, fontsize=9)
    plt.title('Pareto Frontier: Speed vs. Reliability')
    plt.xlabel('Latency (ms) [Lower is Better]')
    plt.ylabel('Success Rate [Higher is Better]')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_frontier.png')
    plt.close()

# =========================================================
# PART B: JSON Validation Analysis (The "New" Logic)
# =========================================================

def analyze_validation_json(json_path: str):
    """
    Analyzes the 'Grand Validation' JSON log.
    Plots distribution of Vectors Fetched vs Success.
    """
    if not os.path.exists(json_path):
        print(f"❌ Error: File not found {json_path}")
        return

    print(f"Reading JSON log: {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    output_dir = os.path.dirname(json_path)
    sns.set_style("whitegrid")
    
    print("\n" + "="*60)
    print("📊 VALIDATION REPORT (JSON)")
    print("="*60)
    
    summary = df.groupby('strategy').agg({
        'target_achieved': 'mean',
        'elapsed_ms': 'mean',
        'vectors_fetched': 'mean',
        'vector_efficiency': 'mean'
    }).sort_values('target_achieved', ascending=False)
    print(summary)
    
    # Validation Plots
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='strategy', y='vectors_fetched')
    plt.title('Distribution of Vectors Fetched (Cost)')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_cost_dist.png'))
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='strategy', y='elapsed_ms')
    plt.title('Latency Stability (Consistency)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_latency_dist.png'))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='strategy', y='target_achieved', errorbar=None)
    plt.title('Reliability (Target Achievement Rate)')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_success.png'))
    
    print(f"\n✅ All validation charts saved to {output_dir}")