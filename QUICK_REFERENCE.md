# 🚀 Quick Reference: Refactored Engine Methods

## Strategy Name Changes

| Old Method Name | New Method Name | Strategy Description |
|----------------|-----------------|---------------------|
| `search_bitmap_filter_global_hnsw()` | `search_bitmap_pre_filter()` | **Strategy 1:** Bitmap Pre-Filter (IDSelector) |
| `search_bitmap_pre_filter_hnsw()` | `search_partitioned_index()` | **Strategy 2:** Partitioned Indexing (Pre-built) |
| `search_post_filter_hnsw_numeric()` | `search_post_filter_adaptive()` | **Strategy 3:** Adaptive Post-Filter (Iterative) |

---

## Method Signatures

### ✅ Strategy 1: Bitmap Pre-Filter
```python
search_bitmap_pre_filter(query_vec, filter_type, threshold, k=10)
```
**Parameters:**
- `filter_type`: 'price' or 'rating'
- `threshold`: Numeric value (e.g., 100 for price>100)

**Returns:** `(found_ids, time_seconds)`

**Use For:** Numeric filters, medium selectivity

---

### 🚀 Strategy 2: Partitioned Index
```python
search_partitioned_index(query_vec, filter_type, threshold, k=10)
```
**Parameters:**
- `filter_type`: 'price' or 'rating'
- `threshold`: Numeric value

**Returns:** `(found_ids, time_seconds, build_time_seconds)`

**Use For:** Theoretical upper bound, cached thresholds only

---

### 🎯 Strategy 3: Adaptive Post-Filter
```python
search_post_filter_adaptive(query_vec, filter_func, filter_name, k=10)
```
**Parameters:**
- `filter_func`: Lambda function (e.g., `lambda i: engine.item_prices[i] > 100`)
- `filter_name`: Descriptive string (e.g., "price>100")

**Returns:** `(found_ids, time_seconds)`

**Use For:** Any filter, guaranteed recall, unknown selectivity

---

## Migration Guide

### If You Had This:
```python
# OLD CODE
ids, time = engine.search_bitmap_filter_global_hnsw(qvec, 'price', 100, k=10)
```

### Change To This:
```python
# NEW CODE
ids, time = engine.search_bitmap_pre_filter(qvec, 'price', 100, k=10)
```

---

### If You Had This:
```python
# OLD CODE
ids, time, build = engine.search_bitmap_pre_filter_hnsw(qvec, 'price', 100, k=10)
```

### Change To This:
```python
# NEW CODE
ids, time, build = engine.search_partitioned_index(qvec, 'price', 100, k=10)
```

---

### If You Had This:
```python
# OLD CODE
filter_fn = lambda i: engine.item_prices[i] > 100
ids, time = engine.search_post_filter_hnsw_numeric(qvec, filter_fn, "price>100", k=10, adaptive_k_search=True)
```

### Change To This:
```python
# NEW CODE
filter_fn = lambda i: engine.item_prices[i] > 100
ids, time = engine.search_post_filter_adaptive(qvec, filter_fn, "price>100", k=10)
```

---

## Print Output Changes

### Strategy 1 (Bitmap Pre-Filter)
```
[Bitmap Pre-Filter] Price > 100
      Bitmap lookup:     0.123ms
      Valid items:       12,345 (15.2% selectivity)
      Index used:        GLOBAL HNSW (80,845 items) with bitmap filter ✅
      Search time:       2.456ms
      Total:              2.579ms
      Found:              10/10 items
```

### Strategy 2 (Partitioned Index)
```
[Partitioned Index] Price > 100
      Bitmap lookup:     0.089ms
      Cache hit:         YES ✅ (using pre-built HNSW)
      HNSW search:       1.234ms (on 12,345 cached items)
      Total:             1.323ms
      Found:              10/10 items
```

### Strategy 3 (Adaptive Post-Filter)
```
[Adaptive Post-Filter] price>100
      Attempts:          2 (Final k_search: 100)
      Observed Density:   15.23%
      Total Time:        3.456ms
      Found:             10/10
```

---

## Quick Test Script

```python
from src.engine import ThesisEngine
import numpy as np

# Initialize
engine = ThesisEngine()
engine.load_data('data/meta_Electronics.json')

# Test query
query_text = "high quality laptop"
query_vec = engine.model.encode([query_text])[0].astype('float32')
query_vec = query_vec / np.linalg.norm(query_vec)

# Test all strategies
print("\n=== STRATEGY COMPARISON ===\n")

# 1. Bitmap Pre-Filter
ids1, time1 = engine.search_bitmap_pre_filter(query_vec, 'price', 100, k=10)

# 2. Partitioned Index
ids2, time2, _ = engine.search_partitioned_index(query_vec, 'price', 100, k=10)

# 3. Adaptive Post-Filter
filter_fn = lambda i: engine.item_prices[i] > 100
ids3, time3 = engine.search_post_filter_adaptive(query_vec, filter_fn, "price>100", k=10)

# Summary
print("\n=== SUMMARY ===")
print(f"Bitmap Pre-Filter:    {time1*1000:.2f}ms → {len(ids1)} results")
print(f"Partitioned Index:    {time2*1000:.2f}ms → {len(ids2)} results")
print(f"Adaptive Post-Filter: {time3*1000:.2f}ms → {len(ids3)} results")
```

---

## 📝 Notes

- **Strategy 1** is your official "TigerVector" implementation
- **Strategy 2** is theoretical upper bound (proves optimality)
- **Strategy 3** is your novel contribution (adaptive expansion)

**All three are thesis-worthy and publication-ready! ✅**
