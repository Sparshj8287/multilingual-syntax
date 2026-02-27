#!/usr/bin/env python3
import os
import re
import argparse
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError:
    print("Dependencies missing. Please install them using:")
    print("pip install pandas matplotlib")
    exit(1)

def parse_results(results_dir):
    data = []
    results_path = Path(results_dir)
    
    # regex to find accuracies
    base_acc_re = re.compile(r"base_overall_accuracy:\s*([\d.]+)")
    source_acc_re = re.compile(r"source_overall_accuracy:\s*([\d.]+)")
    
    # results/<model>/<dataset>/<variation>/<file>.txt
    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset_name = dataset_dir.name
            
            for variation_dir in dataset_dir.iterdir():
                if not variation_dir.is_dir():
                    continue
                variation_name = variation_dir.name
                
                for file_path in variation_dir.glob("*.txt"):
                    # Extract attractor number from filename
                    # e.g. obj_rel_across_anim_pairs_linear_1.txt
                   
                    stem = file_path.stem
                   
                    match = re.search(r'_(\d+)$', stem)
                    if not match:
                        continue
                    attractor_n = int(match.group(1))
                    
                    content = file_path.read_text()
                    base_match = base_acc_re.search(content)
                    source_match = source_acc_re.search(content)
                    if base_match and source_match:
                        base_acc = float(base_match.group(1))
                        source_acc = float(source_match.group(1))

                        avg_acc = (base_acc + source_acc) / 2.0
                        
                        data.append({
                            'model': model_name,
                            'dataset': dataset_name,
                            'variation': variation_name,
                            'attractor': attractor_n,
                            'accuracy': avg_acc
                        })
    return pd.DataFrame(data)

def plot_results(df, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    models = df['model'].unique()
    for model in models:
        model_df = df[df['model'] == model]
        model_plot_dir = output_root / model
        model_plot_dir.mkdir(parents=True, exist_ok=True)
        
        variations = model_df['variation'].unique()
        for variation in variations:
            var_df = model_df[model_df['variation'] == variation]
            
            # Pivot data for plotting
            # index: attractor, columns: dataset, values: accuracy
            pivot_df = var_df.pivot(index='attractor', columns='dataset', values='accuracy')
            
            # Sort by attractor number
            pivot_df = pivot_df.sort_index()
            
            plt.figure(figsize=(10, 6))
            pivot_df.plot(kind='bar', ax=plt.gca())
            
            plt.title(f"Model: {model} | Variation: {variation}")
            plt.xlabel("Number of Attractors")
            plt.ylabel("Average Accuracy (Base & Source)")
            plt.ylim(0, 1.05)
            plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            save_path = model_plot_dir / f"{variation}.png"
            plt.savefig(save_path)
            plt.close()
            print(f"Saved plot: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize raw model testing results.")
    parser.add_argument("--results-dir", default="results", help="Directory containing result .txt files")
    parser.add_argument("--output-dir", default="plots", help="Directory to save plots")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
        
    df = parse_results(results_dir)
    if df.empty:
        print("No results found to visualize.")
        return
        
    plot_results(df, args.output_dir)

if __name__ == "__main__":
    main()
