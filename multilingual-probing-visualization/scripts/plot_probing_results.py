import os
import glob
import re
import argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Plot probing results.')
    parser.add_argument('--metric_file', type=str, default='test.uuas', help='Name of the metric file to read (e.g., test.uuas, dev.spearmanr)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../experiments'))

    for model_name in os.listdir(experiments_dir):
        for task_name in os.listdir(os.path.join(experiments_dir, model_name)):
            base_dir = os.path.join(experiments_dir, model_name, task_name)
            base_dir = os.path.abspath(base_dir)
            model_name = base_dir.split('/')[-2]
            task_name = base_dir.split('/')[-1]
            output_dir = os.path.join(script_dir, '../visualizations_plots', model_name, task_name, "polar_results")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)



            output_plot_name= f'{model_name}_{task_name}_{args.metric_file}.png'
 


            print(f"Searching in: {base_dir}")

            data = []

            subdirs = glob.glob(os.path.join(base_dir, '*'))
            
            for subdir in subdirs:
                if not os.path.isdir(subdir):
                    continue

                config_files = glob.glob(os.path.join(subdir, 'temp_config_layer_*.yaml'))
                if not config_files:
                    continue
                
                config_file_path = config_files[0]
                config_filename = os.path.basename(config_file_path)
                
                match = re.search(r'temp_config_layer_(\d+)\.yaml', config_filename)
                if not match:
                    print(f"Could not extract layer number from {config_filename} in {subdir}, skipping.")
                    continue
                
                layer_num = int(match.group(1))

                metric_path = os.path.join(subdir, 'polar_results', args.metric_file)
                if not os.path.exists(metric_path):
                    # Check in polar_results subdirectory
                    metric_path = os.path.join(subdir, args.metric_file)
                    
                if not os.path.exists(metric_path):
                    print(f"Metric file {args.metric_file} not found in {subdir} or {subdir}/polar_results, skipping.")
                    continue

                try:
                    with open(metric_path, 'r') as f:
                        content = f.read().strip()
                        # Parse float
                        val = float(content)
                        data.append((layer_num, val))
                except ValueError:
                    print(f"Could not parse value from {metric_path} (content: '{content[:20]}...'), skipping.")
                except Exception as e:
                    print(f"Error reading {metric_path}: {e}")

            if not data:
                print("No data found to plot.")
                return

            data.sort(key=lambda x: x[0])
            
            layers = [x[0] for x in data]
            values = [x[1] for x in data]

            print(f"Found {len(data)} data points.")
            for l, v in zip(layers, values):
                print(f"Layer {l}: {v}")


            plt.figure(figsize=(12, 6))
            plt.plot(layers, values, marker='o', linestyle='-', color='b')
            plt.title(f'Layer vs {args.metric_file} for {model_name} on {task_name}')
            plt.xlabel('Layer Number')
            plt.ylabel(args.metric_file)
            plt.grid(True)
            

            if len(layers) < 30:
                plt.xticks(layers)

            output_path = os.path.join(output_dir, output_plot_name)
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    main()
