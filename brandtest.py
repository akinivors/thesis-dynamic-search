import json
from collections import Counter

def find_heavy_hitters():
    print("Scanning dataset for massive groups...")
    brand_counts = Counter()
    
    # We will also scan 'main_cat' if brands aren't big enough
    cat_counts = Counter() 
    
    with open('data/meta_Electronics.json', 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'brand' in data:
                    brand_counts[data['brand']] += 1
                if 'main_cat' in data:
                    cat_counts[data['main_cat']] += 1
            except: continue
            
    print("\n--- TOP 5 BRANDS ---")
    for b, c in brand_counts.most_common(5):
        print(f"Brand: '{b}' | Count: {c}")
        
    print("\n--- TOP 5 CATEGORIES ---")
    for c, n in cat_counts.most_common(5):
        print(f"Category: '{c}' | Count: {n}")

if __name__ == "__main__":
    find_heavy_hitters()