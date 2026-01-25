import os
import re
import argparse
import matplotlib.pyplot as plt


def _find_run_roots(experiments_dir):
    run_roots = []
    for root, dirs, _ in os.walk(experiments_dir):
        layer_dirs = [d for d in dirs if re.match(r'^layer-\d+$', d)]
        model_layer_dirs = [d for d in dirs if re.match(r'^model-layer-\d+-', d)]
        if layer_dirs:
            run_roots.append((root, layer_dirs))
            dirs[:] = [d for d in dirs if d not in layer_dirs]
        elif model_layer_dirs:
            run_roots.append((root, model_layer_dirs))
            dirs[:] = [d for d in dirs if d not in model_layer_dirs]
    return run_roots


def _parse_layer_from_dir(dirname):
    match = re.search(r'layer-(\d+)', dirname)
    if match:
        return int(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(description='Plot probing results.')
    parser.add_argument('--metric_file', type=str, default='test.uuas',
                        help='Name of the metric file to read (e.g., test.uuas)')
    parser.add_argument('--experiments_dir', type=str, default='',
                        help='Root directory containing experiment outputs')
    parser.add_argument('--plots_dir', type=str, default='',
                        help='Output directory for plots')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = args.experiments_dir or os.path.abspath(os.path.join(script_dir, '../experiments'))
    plots_dir = args.plots_dir or os.path.abspath(os.path.join(script_dir, '../visualizations_plots'))

    run_roots = _find_run_roots(experiments_dir)
    if not run_roots:
        print(f"No experiment runs found under {experiments_dir}")
        return

    for run_root, layer_dirs in run_roots:
        print(f"Searching in: {run_root}")
        data = []
        for layer_dir in layer_dirs:
            layer_num = _parse_layer_from_dir(layer_dir)
            if layer_num is None:
                continue
            layer_path = os.path.join(run_root, layer_dir)
            metric_path = os.path.join(layer_path, 'polar_results', args.metric_file)
            if not os.path.exists(metric_path):
                metric_path = os.path.join(layer_path, args.metric_file)
            if not os.path.exists(metric_path):
                continue
            try:
                with open(metric_path, 'r') as f:
                    content = f.read().strip()
                    val = float(content)
                    data.append((layer_num, val))
            except ValueError:
                print(f"Could not parse value from {metric_path} (content: '{content[:20]}...'), skipping.")
            except Exception as e:
                print(f"Error reading {metric_path}: {e}")

        if not data:
            print("No data found to plot for this run.")
            continue

        data.sort(key=lambda x: x[0])
        layers = [x[0] for x in data]
        values = [x[1] for x in data]

        rel_path = os.path.relpath(run_root, experiments_dir)
        output_dir = os.path.join(plots_dir, rel_path)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{args.metric_file}.png")

        plt.figure(figsize=(12, 6))
        plt.plot(layers, values, marker='o', linestyle='-', color='b')
        plt.title(f'Layer vs {args.metric_file} for {rel_path}')
        plt.xlabel('Layer Number')
        plt.ylabel(args.metric_file)
        plt.grid(True)
        if len(layers) < 30:
            plt.xticks(layers)
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")


if __name__ == '__main__':
    main()
