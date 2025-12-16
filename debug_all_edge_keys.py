import json
import os

# --- Configuration (MUST MATCH your experiment setup) ---
DATA_FILE_NAME = "meta_Electronics.json"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', DATA_FILE_NAME) 

# --- Keys to Check ---
COMMON_EDGE_KEYS = [
    'related',           # Key used in the previous fix attempt
    'bought_together',   # Key that was empty
    'also_buy',
    'also_view',
    'buy_after_viewing',
    'frequently_bought_together' # A key sometimes seen in this dataset
]

def debug_all_edge_keys(filepath, max_prints=5):
    """
    Reads the file and prints the raw value of any known EDGE_KEY whenever it is found,
    until max_prints is reached for each key.
    """
    print("=" * 70)
    print(f"🔬 DEBUGGING MULTIPLE EDGE KEYS IN: '{DATA_FILE_NAME}'")
    print("=" * 70)
    
    # Path Fallback Check
    if not os.path.exists(filepath):
        # Fallback path logic remains here...
        fallback_path = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'data', DATA_FILE_NAME)
        if os.path.exists(fallback_path):
             filepath = fallback_path
        else:
            print(f"❌ ERROR: File not found at the expected path: {filepath}")
            return
            
    key_found_count = {key: 0 for key in COMMON_EDGE_KEYS}

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            
            try:
                data = json.loads(line)
                
                for key in COMMON_EDGE_KEYS:
                    if key in data:
                        edge_value = data.get(key)
                        
                        # Only print if we haven't hit the limit for this key AND the value is usable
                        if key_found_count[key] < max_prints and edge_value is not None and edge_value != [] and (isinstance(edge_value, list) or isinstance(edge_value, dict)):
                            
                            value_type = type(edge_value)
                            value_len = len(edge_value) if isinstance(edge_value, (list, dict)) else 0
                            
                            # Print a useful representation
                            print(f"✅ Line {i+1}: Found Key '{key}'")
                            print(f"   Type: {value_type}, Length: {value_len}")
                            print(f"   Sample: {str(edge_value)[:100]}...")
                            
                            key_found_count[key] += 1
                            
            except json.JSONDecodeError:
                continue

    print("\n" + "=" * 70)
    print("SUMMARY OF KEY USABILITY:")
    print("=" * 70)
    for key, count in key_found_count.items():
        if count > 0:
            print(f"🟢 FOUND Usable data for '{key}' ({count} times in sample).")
        else:
            print(f"🔴 NOT FOUND Usable data for '{key}'.")

if __name__ == "__main__":
    debug_all_edge_keys(DATA_PATH)