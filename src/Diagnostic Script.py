import json
from collections import Counter

FILE_PATH = "meta_Electronics.json"
LIMIT = 20000

print(f"Scanning first {LIMIT} items for Brand statistics...")

brands = []
count = 0

with open(FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            if 'brand' in data:
                brands.append(data['brand'])
            count += 1
            if count >= LIMIT:
                break
        except: continue

# 1. Total Distribution
c = Counter(brands)
total_brands = len(c)
print(f"\nTotal Unique Brands: {total_brands}")

# 2. Check Specific Targets
target_rare = "Kensington"
target_common = "Samsung"

print(f"\n--- DIAGNOSTICS ---")
print(f"Count of '{target_rare}': {c[target_rare]}")
print(f"Count of '{target_common}': {c[target_common]}")

# 3. Find Better Candidates?
print(f"\nTop 10 Most Frequent Brands in this chunk:")
for b, cnt in c.most_common(10):
    print(f"  {b}: {cnt}")

print("\n(Use one of these Top 10 as your 'Common' brand for the next reliable test)")