import json
import numpy as np
import collections
import statistics

def inspect_dataset(filepath):
    print(f"=== INSPECTING DATASET: {filepath} ===")
    
    # Counters
    total_items = 0
    price_counts = []
    rating_counts = []
    brands = collections.Counter()
    categories = collections.Counter()
    keys_histogram = collections.Counter()
    
    # 1. SCAN THE FILE
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                total_items += 1
                
                # Check available keys
                for k in data.keys():
                    keys_histogram[k] += 1
                
                # Extract Price
                # (Handle formats like "$19.99" or "19.99")
                p_raw = data.get('price', '')
                if p_raw:
                    try:
                        p_str = str(p_raw).replace('$', '').replace(',', '')
                        p_val = float(p_str)
                        if p_val > 0:
                            price_counts.append(p_val)
                    except:
                        pass
                
                # Extract Rating
                r_raw = data.get('average_rating', 0)
                if r_raw:
                    try:
                        rating_counts.append(float(r_raw))
                    except:
                        pass
                        
                # Extract Brand & Category
                b = data.get('brand', 'Unknown')
                brands[b] += 1
                
                c = data.get('main_cat', 'Unknown')
                categories[c] += 1
                
                if total_items % 100000 == 0:
                    print(f"   ...scanned {total_items} items...")
                    
            except json.JSONDecodeError:
                continue

    # 2. ANALYZE & PRINT
    print("\n=== DATASET STATISTICS ===")
    print(f"Total Products: {total_items:,}")
    
    # Keys
    print("\n[Column Presence]")
    for k, v in keys_histogram.most_common(10):
        print(f"   - {k}: {v:,} ({(v/total_items)*100:.1f}%)")

    # Prices
    if price_counts:
        price_arr = np.array(price_counts)
        print("\n[Price Distribution]")
        print(f"   - Count: {len(price_arr):,} ({(len(price_arr)/total_items)*100:.1f}% coverage)")
        print(f"   - Min: ${np.min(price_arr):.2f}")
        print(f"   - Max: ${np.max(price_arr):.2f}")
        print(f"   - Median: ${np.median(price_arr):.2f}")
        print(f"   - Mean:   ${np.mean(price_arr):.2f}")
        
        # Calculate Selectivity Buckets for Experiments
        print("\n   [Experimental Thresholds - Selectivity]")
        for pct in [10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(price_arr, pct)
            # Inverse: items > val
            count_above = np.sum(price_arr > val)
            sel_pct = (count_above / total_items) * 100
            print(f"      To test {100-pct}% selectivity (Top {100-pct}% items): Filter Price > ${val:.2f} ({sel_pct:.1f}% actual)")

    # Ratings
    if rating_counts:
        print("\n[Rating Distribution]")
        r_counter = collections.Counter(rating_counts)
        for r in sorted(r_counter.keys()):
            print(f"   - {r} Stars: {r_counter[r]:,} items")

    # Brands (Categorical Test)
    print("\n[Top Brands - For 'Rare' vs 'Common' Tests]")
    for b, count in brands.most_common(10):
        print(f"   - {b}: {count:,} ({(count/total_items)*100:.2f}%)")
        
    print("\n[Top Categories]")
    for c, count in categories.most_common(10):
        print(f"   - {c}: {count:,}")

if __name__ == "__main__":
    # REPLACE THIS WITH YOUR ACTUAL PATH
    dataset_path = 'data/meta_Electronics.json'
    inspect_dataset(dataset_path)