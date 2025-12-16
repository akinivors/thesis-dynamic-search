import json
from collections import Counter

def scan_distribution():
    print("Scanning dataset distribution...")
    
    # Store counts for everything (Brands + Categories)
    counts = Counter()
    
    with open('data/meta_Electronics.json', 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # We track both, labeling them so we know which is which
                if data.get('brand'):
                    counts[f"[Brand] {data['brand']}"] += 1
                if data.get('main_cat'):
                    counts[f"[Category] {data['main_cat']}"] += 1
            except: continue

    # Define our "Zones" of interest based on your previous threshold (~18k)
    zones = {
        "MASSIVE (> 100k)": [],
        "HIGH (20k - 100k) - [Post-Filter Territory]": [],
        "BOUNDARY (15k - 20k) - [The Edge Case]": [],
        "MEDIUM (5k - 15k) - [Pre-Filter Territory]": [],
        "SMALL (< 100)": []
    }

    # Sort everything by count descending
    sorted_items = counts.most_common()

    for name, count in sorted_items:
        if count > 100000:
            zones["MASSIVE (> 100k)"].append((name, count))
        elif 20000 <= count <= 100000:
            zones["HIGH (20k - 100k) - [Post-Filter Territory]"].append((name, count))
        elif 15000 <= count < 20000:
            zones["BOUNDARY (15k - 20k) - [The Edge Case]"].append((name, count))
        elif 5000 <= count < 15000:
            zones["MEDIUM (5k - 15k) - [Pre-Filter Territory]"].append((name, count))
        elif count < 100:
            # Just keep a few small ones, we have plenty
            if len(zones["SMALL (< 100)"]) < 5:
                zones["SMALL (< 100)"].append((name, count))

    # --- REPORTING ---
    print("\n" + "="*60)
    print("DATASET DISTRIBUTION REPORT")
    print("="*60)

    for zone, items in zones.items():
        print(f"\n>>> {zone}")
        if not items:
            print("    (No items found in this range)")
        else:
            # Print top 5 and bottom 5 of this zone to give variety
            sample = items[:5]
            for name, count in sample:
                print(f"    {count:<8} | {name}")
            
            if len(items) > 5:
                print("    ...")

if __name__ == "__main__":
    scan_distribution()