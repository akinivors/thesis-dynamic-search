import json

def show_keys():
    print("Scanning 'meta_Electronics.json' for schema...")
    with open('meta_Electronics.json', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                # Print the keys of the first valid JSON object
                print(f"\n[Line {i+1} Keys]:")
                print(list(data.keys()))
                
                # Check for "also_buy" or similar variants
                if 'also_buy' in data:
                    print(f"   -> FOUND 'also_buy': {data['also_buy'][:3]}")
                if 'also_view' in data:
                    print(f"   -> FOUND 'also_view': {data['also_view'][:3]}")
                    
                return # Stop after the first one
            except: continue

if __name__ == "__main__":
    show_keys()