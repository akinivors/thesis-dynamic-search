# Experiments Organization

This directory contains all experimental scripts and their results, organized by quality and purpose.

## 📁 Directory Structure

### `thesis_final/` ✅
**Well-designed, publication-worthy experiments for the final thesis**

| Script | Purpose | Key Metrics | Results Location |
|--------|---------|-------------|------------------|
| `run_deep_dive.py` | Matrix analysis across categories/brands with ground truth | Recall, Time, Quality | `results/thesis_final/deep_dive_analysis.json` |
| `run_tiger_quality.py` | Quality-focused comparison with recall metrics | Recall vs Ground Truth | `results/thesis_final/tiger_quality_analysis.json` |
| `run_correct_comparison.py` | Three-way: TRUE Bitmap vs Post-Filter OLD vs FIXED | Time, Recall, Correctness | `results/thesis_final/correct_comparison_*.json` |
| `run_ultra_detailed_comparison.py` | Product-level analysis with distance statistics | Distance stats, Overlap | `results/thesis_final/ultra_detailed_*.json` |
| `run_selectivity_sweep.py` | Systematic selectivity sweep (0.05%-12%) | Crossover points | `results/thesis_final/selectivity_sweep_*.json` |
| `run_final_adaptive.py` | Validates adaptive optimizer's decision-making | Cost model predictions | Console output only |

**Why these are good:**
- Clear research questions
- Proper ground truth comparison
- Recall metrics included
- Systematic methodology
- Reproducible results
- Publication-ready quality

---

### `deprecated_early/` ⚠️
**Early iterations, redundant experiments, or debugging artifacts**

| Script | Issue | Superseded By |
|--------|-------|---------------|
| `run_comprehensive_experiment.py` | Mixes cached/non-cached confusingly | `run_selectivity_sweep.py` |
| `run_detailed_analysis.py` | Overlaps with better experiments | `run_deep_dive.py`, `run_tiger_quality.py` |
| `run_detailed_comparison.py` | Shows distances but no recall | `run_ultra_detailed_comparison.py` |
| `run_recall_fix_experiment.py` | Debugging artifact (tests OLD vs FIXED) | `run_correct_comparison.py` |
| `run_true_bitmap_test.py` | Implementation test, not research | `run_tiger_quality.py` |
| `run_tiger_battle.py` | Only time metrics, no recall/quality | `run_tiger_quality.py` |

**Why deprecated:**
- Early prototypes superseded by better versions
- Debugging/fixing artifacts (not research contributions)
- Redundant with more comprehensive experiments
- Missing key metrics (recall, ground truth)

**Keep for:** Showing evolution of research methodology in thesis appendix

---

### `archive_obsolete/` ❌
**Tests implementation bugs or obvious things - minimal research value**

| Script | Problem |
|--------|---------|
| `run_hardcoding_test.py` | Tests implementation limitation (hardcoded cache keys) - engineering issue, not research |
| `run_numeric_ultimate_showdown.py` | Proves obvious fact (categorical ≠ numeric) - strawman experiment |

**Why obsolete:**
- Tests known facts or implementation bugs
- No research contribution
- Should not be in thesis

**Keep for:** Historical record only

---

### `situation_3&4/`, `situation_5/` 📦
**Older experimental setups - kept for historical reference**

These contain earlier versions of experiments before the final methodology was established.

---

## 🎯 Quick Decision Guide

**For thesis writing:**
- Use ONLY `thesis_final/` experiments
- These have proper methodology and metrics

**For showing research evolution:**
- Mention `deprecated_early/` in appendix
- Show how methodology improved over time

**For code review/debugging:**
- Check `archive_obsolete/` if needed
- Don't reference in thesis

---

## 📊 Results Organization

Results are organized in parallel structure:

```
results/
├── thesis_final/          # Final results for thesis
│   ├── deep_dive_analysis.json
│   ├── tiger_quality_analysis.json
│   ├── matrix_deep_dive_fixed.json
│   ├── matrix_detailed_products_fixed.json
│   ├── correct_comparison_*.json
│   ├── ultra_detailed_*.json
│   └── selectivity_sweep_*.json
│
└── deprecated_early/      # Early iteration results
    ├── experiment_results_*.json/log
    ├── detailed_analysis_*.json/log
    ├── detailed_comparison_*.json/log
    ├── recall_fix_*.json/log
    └── matrix_deep_dive.json (older versions)
```

---

## 🔬 Research Questions Addressed

### thesis_final/ experiments answer:

1. **Performance vs Selectivity** (`run_selectivity_sweep.py`)
   - What's the crossover point between strategies?
   - How does selectivity affect strategy choice?

2. **Quality Metrics** (`run_tiger_quality.py`, `run_deep_dive.py`)
   - What's the recall for each strategy?
   - How close are results to ground truth?

3. **Product-Level Analysis** (`run_ultra_detailed_comparison.py`)
   - Which specific products differ between strategies?
   - Distance score distributions?

4. **Adaptive Optimization** (`run_final_adaptive.py`)
   - Does the cost model predict correctly?
   - When does the optimizer choose each strategy?

5. **Implementation Correctness** (`run_correct_comparison.py`)
   - Are all implementations correct?
   - Fixed vs OLD post-filter comparison

---

## 📝 Last Updated
December 20, 2025

## 👤 Maintainer
Akin Akdogan - Thesis Research Project
