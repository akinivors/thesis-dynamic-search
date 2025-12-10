import json

def inspect_structure():
    print("Scanning 'meta_Electronics.json' for structure...")
    
    with open("meta_Electronics.json", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                
                # Check for ANY field that looks like relationships
                # Standard keys usually include 'related', 'also_bought', etc.
                if 'related' in data:
                    print(f"\n[FOUND 'related' field on line {i+1}]")
                    print(json.dumps(data['related'], indent=4))
                    return
                
                # Sometimes it's not nested under 'related'
                if 'also_bought' in data:
                    print(f"\n[FOUND 'also_bought' at root on line {i+1}]")
                    print(f"Sample: {data['also_bought']}")
                    return
                    
            except: continue
            
            if i > 100000:
                print("Scanned 100,000 lines and found NO relationship fields.")
                print("Your dataset version might be missing edge data entirely.")
                return

if __name__ == "__main__":
    inspect_structure()