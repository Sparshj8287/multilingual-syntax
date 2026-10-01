import yaml
import glob
import os

config_dir = "multilingual-probing-visualization/configs/syntax_boundless/"
for filepath in glob.glob(os.path.join(config_dir, "*.yaml")):
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    
    if "selector" in data.get("dataset", {}):
        del data["dataset"]["selector"]
    
    data["dataset"]["corpus"] = {
        "root": "datasets/en",
        "train_path": "train.conllu",
        "dev_path": "dev.conllu",
        "test_path": "test.conllu"
    }

    # Optionally update the reporting path to distinguish
    if "reporting" in data:
        data["reporting"]["root"] = "experiments/syntax_boundless_en"

    # Also make sure use_disk is True, which is typically for cached embeddings
    # Wait, they are running on raw texts and transformer-lens, so use_disk isn't for embeddings here.
    # Ah, the model config for transformer-lens might read conllu files and run the model on them.
    
    with open(filepath, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"Updated {filepath}")
