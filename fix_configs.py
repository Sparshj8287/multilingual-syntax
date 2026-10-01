import yaml
import glob
import os

config_dir = "multilingual-probing-visualization/configs/syntax_boundless/"
for filepath in glob.glob(os.path.join(config_dir, "*.yaml")):
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    
    if "reporting" in data and "layout" in data["reporting"]:
        del data["reporting"]["layout"]
    
    # Also I need to ensure there are no other missing keys
    
    with open(filepath, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"Fixed {filepath}")
