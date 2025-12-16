import json

DATA_PATH = "data/meta_Electronics.json"

def check_structure():
    print(f"--- INSPECTING {DATA_PATH} ---")
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5: break
                try:
                    data = json.loads(line)
                    print(f"\n[Item {i}] Keys Found: {list(data.keys())}")
                    
                    # Check our expected keys specifically
                    print(f"   -> 'asin': {data.get('asin')}")
                    print(f"   -> 'title': {str(data.get('title'))[:30]}...")
                    print(f"   -> 'brand': {data.get('brand')}")
                    print(f"   -> 'also_buy': {data.get('also_buy')}")
                    
                except json.JSONDecodeError:
                    print(f"\n[Item {i}] Failed to decode JSON")
    except FileNotFoundError:
        print("Error: File not found. Check path.")

if __name__ == "__main__":
    check_structure()