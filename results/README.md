# Results Organization

This directory contains all experimental results, organized by quality and usage.

## 📁 Directory Structure

### `thesis_final/` ✅
**Publication-ready results for the final thesis**

| File | Source Experiment | Description |
|------|-------------------|-------------|
| `deep_dive_analysis.json` | `run_deep_dive.py` | Matrix analysis: strategies vs categories/brands (k=10, k=50) |
| `tiger_quality_analysis.json` | `run_tiger_quality.py` | Quality comparison with recall metrics vs ground truth |
| `matrix_deep_dive_fixed.json` | `run_deep_dive.py` | Fixed version of matrix analysis |
| `matrix_detailed_products_fixed.json` | `run_deep_dive.py` | Product-level details for matrix analysis |
| `correct_comparison_*.json` | `run_correct_comparison.py` | Three-way comparison (Bitmap/Post-OLD/Post-FIXED) |
| `ultra_detailed_*.json` | `run_ultra_detailed_comparison.py` | Ultra-detailed product-level analysis with distance stats |
| `selectivity_sweep_*.json` | `run_selectivity_sweep.py` | Selectivity sweep from 0.05% to 12% |
| `selectivity_sweep_*.log` | `run_selectivity_sweep.py` | Detailed logs of selectivity experiments |

**Use these for:**
- Thesis figures and tables
- Performance comparisons
- Quality metrics (recall)
- Final conclusions

---

### `deprecated_early/` ⚠️
**Early iteration results - kept for reference**

| File Pattern | Source Experiment | Status |
|--------------|-------------------|--------|
| `experiment_results_*.json/log` | `run_comprehensive_experiment.py` | Superseded by selectivity_sweep |
| `detailed_analysis_*.json/log` | `run_detailed_analysis.py` | Superseded by deep_dive & tiger_quality |
| `detailed_comparison_*.json/log` | `run_detailed_comparison.py` | Superseded by ultra_detailed_comparison |
| `recall_fix_*.json/log` | `run_recall_fix_experiment.py` | Debugging artifact |
| `matrix_deep_dive.json` | `run_deep_dive.py` | Old version (use _fixed instead) |
| `matrix_deep_dive1.json` | `run_deep_dive.py` | Old version (use _fixed instead) |
| `matrix_detailed_products.json` | `run_deep_dive.py` | Old version (use _fixed instead) |

**Keep for:**
- Showing methodology evolution in appendix
- Historical reference
- Understanding how experiments improved

**Don't use for:**
- Final thesis results
- Performance comparisons in main text

---

## 📊 Key Result Files Explained

### `deep_dive_analysis.json`
**Purpose:** Comprehensive comparison across different filter types and sizes

**Structure:**
```json
[
  {
    "k": 10,
    "target": "Computers",
    "count": 244659,
    "query": "high performance desktop",
    "ground_truth": { "time_ms": 559.7, "top_results": [...] },
    "strategies": {
      "Flat Post": { "time_ms": 187.66, "recall": 1.0, "top_results": [...] },
      "HNSW Post": { "time_ms": 5.46, "recall": 1.0, "top_results": [...] },
      "HNSW Bitmap": { "time_ms": 21.24, "recall": 0.8, "top_results": [...] }
    }
  }
]
```

**Key Insights:**
- Tests across different selectivities (6 items to 244k items)
- Includes recall metrics vs ground truth
- Shows k=10 and k=50 results
- Demonstrates strategy performance trade-offs

---

### `tiger_quality_analysis.json`
**Purpose:** Focus on quality/recall metrics for each strategy

**Contains:**
- Ground truth results (100% recall by definition)
- Recall percentage for each strategy
- Top result samples for verification
- Time performance

**Use for:** Proving that strategies maintain quality while improving speed

---

### `selectivity_sweep_*.json`
**Purpose:** Find exact crossover points between strategies

**Covers:** 24 different selectivity points from 0.05% to 12%

**Key Finding:** Crossover point for strategy selection

---

## 🎯 Quick Reference

### For Thesis Figures
Use: `thesis_final/*.json`

### For Methodology Evolution
Reference: `deprecated_early/*.json` (appendix only)

### For Performance Tables
Primary: `deep_dive_analysis.json`, `tiger_quality_analysis.json`
Secondary: `selectivity_sweep_*.json` for detailed selectivity analysis

### For Quality Metrics
Best: `tiger_quality_analysis.json` (has explicit recall measurements)
Also: `deep_dive_analysis.json` (has recall for each test case)

---

## 🔍 Result File Naming Convention

- `*_20251217_*.json` = December 17, 2025 experiments
- `*_20251218_*.json` = December 18, 2025 experiments  
- `*_20251220_*.json` = December 20, 2025 experiments (latest)
- `*_fixed.json` = Fixed version (use these over non-fixed)

---

## 📝 Last Updated
December 20, 2025

## 👤 Maintainer
Akin Akdogan - Thesis Research Project
