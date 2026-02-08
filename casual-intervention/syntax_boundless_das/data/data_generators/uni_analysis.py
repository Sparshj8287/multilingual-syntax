import os

def analyze_unimorph_features(file_path):
    unique_features = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) < 3:
                    print(f"Malformed line: {line}")
                    # Sometimes files might have empty lines or malformed lines
                    continue
                
                features_str = parts[2]
                if features_str:
                    unique_features.add(features_str)
                # features = features_str.split(';')
                
                # for feature in features:
                #     if feature: # Avoid empty strings if any
                #         unique_features.add(feature)
                        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    print(f"Total unique features found: {len(unique_features)}")
    print("-" * 30)
    print("Unique Features:")
    for feature in sorted(unique_features):
        print(feature)

if __name__ == "__main__":
    # Path relative to this script: ../eng/eng
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(script_dir, "../eng/eng")
    
    print(f"Analyzing file: {data_file_path}")
    analyze_unimorph_features(data_file_path)
