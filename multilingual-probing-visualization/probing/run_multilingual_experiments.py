"""Run multilingual probe experiments across language splits."""

from argparse import ArgumentParser
import copy
import os
import subprocess
import sys
import yaml


EXPERIMENT_TYPES = {'in_lang', 'single_train', 'holdout', 'all_lang'}
DEFAULT_EXPERIMENT_TYPES = ['in_lang', 'single_train', 'holdout', 'all_lang']


def _slug_model_tag(args, exp_cfg):
  if exp_cfg.get('model_tag'):
    return str(exp_cfg['model_tag'])
  if 'metadata' in args and args['metadata'].get('model_tag'):
    return str(args['metadata']['model_tag'])
  decoder_cfg = args.get('decoder_model', {})
  model_name = decoder_cfg.get('model_name') or args.get('model', {}).get('model_name')
  if model_name:
    return str(model_name).replace('/', '-')
  reporting_root = args.get('reporting', {}).get('root')
  if reporting_root:
    return os.path.basename(reporting_root.rstrip('/'))
  return 'model'


def _discover_languages(root_path, split_paths):
  languages = []
  if not os.path.exists(root_path):
    return languages
  for entry in sorted(os.listdir(root_path)):
    lang_dir = os.path.join(root_path, entry)
    if not os.path.isdir(lang_dir):
      continue
    has_any = any(os.path.exists(os.path.join(lang_dir, split_path)) for split_path in split_paths.values())
    if has_any:
      languages.append(entry)
  return languages


def _format_train_spec(train_keys, all_langs, eval_lang):
  if set(train_keys) == set(all_langs):
    return "train-all"
  if eval_lang not in train_keys and len(train_keys) == len(all_langs) - 1:
    return f"train-all-except-{eval_lang}"
  if len(train_keys) == 1:
    return f"train-{train_keys[0]}"
  return "train-" + "+".join(train_keys)


def _safe_component(value):
  return str(value).replace(os.sep, '-').replace(' ', '-')


def _write_yaml(path, payload):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'w') as fout:
    yaml.safe_dump(payload, fout, sort_keys=False)


def _run_experiment(config_path, results_dir, train_probe, report_results):
  run_experiment_path = os.path.join(os.path.dirname(__file__), 'run_experiment.py')
  cmd = [
      sys.executable,
      run_experiment_path,
      config_path,
      '--results-dir',
      results_dir,
      '--train-probe',
      str(train_probe),
      '--report-results',
      str(report_results),
  ]
  subprocess.run(cmd, check=True)


def _build_runs(experiment_type, languages):
  runs = []
  if experiment_type == 'in_lang':
    for lang in languages:
      runs.append((lang, [lang]))
  elif experiment_type == 'single_train':
    for train_lang in languages:
      for eval_lang in languages:
        if eval_lang == train_lang:
          continue
        runs.append((eval_lang, [train_lang]))
  elif experiment_type == 'holdout':
    for eval_lang in languages:
      train_keys = [lang for lang in languages if lang != eval_lang]
      runs.append((eval_lang, train_keys))
  elif experiment_type == 'all_lang':
    for eval_lang in languages:
      runs.append((eval_lang, list(languages)))
  else:
    raise ValueError(f"Unknown experiment type: {experiment_type}")
  return runs


def main():
  parser = ArgumentParser(description="Run multilingual probe experiments.")
  parser.add_argument('--config', required=True, help='Path to experiment config yaml')
  parser.add_argument('--experiments', default='', help='Comma-separated experiment types')
  parser.add_argument('--languages', default='', help='Comma-separated language codes')
  parser.add_argument('--results-root', default='', help='Root directory for experiment outputs')
  parser.add_argument('--configs-root', default='', help='Root directory for generated configs')
  parser.add_argument('--write-configs', action='store_true', help='Write per-run configs')
  parser.add_argument('--no-write-configs', action='store_true', help='Disable config generation')
  parser.add_argument('--train-probe', type=int, default=1, help='Train probe (1/0)')
  parser.add_argument('--report-results', type=int, default=1, help='Report results (1/0)')
  parser.add_argument('--layer', type=int, default=None, help='Override model layer')
  parser.add_argument('--dry-run', action='store_true', help='Print planned runs only')
  cli_args = parser.parse_args()

  with open(cli_args.config, 'r') as config_file:
    base_args = yaml.safe_load(config_file)

  exp_cfg = base_args.get('multilingual_experiments', {})

  experiment_types = []
  if cli_args.experiments:
    experiment_types = [x.strip() for x in cli_args.experiments.split(',') if x.strip()]
  else:
    experiment_types = exp_cfg.get('types', list(DEFAULT_EXPERIMENT_TYPES))
  experiment_types = [x for x in experiment_types if x in EXPERIMENT_TYPES]
  if not experiment_types:
    raise ValueError("No valid experiment types provided.")

  dataset_cfg = base_args.get('dataset', {})
  split_paths = {
      'train': dataset_cfg.get('corpus', {}).get('train_path', 'train.conllu'),
      'dev': dataset_cfg.get('corpus', {}).get('dev_path', 'dev.conllu'),
      'test': dataset_cfg.get('corpus', {}).get('test_path', 'test.conllu'),
  }
  if cli_args.languages:
    languages = [x.strip() for x in cli_args.languages.split(',') if x.strip()]
  else:
    languages = exp_cfg.get('languages') or _discover_languages(
        dataset_cfg.get('corpus', {}).get('root', 'datasets'), split_paths)
  if not languages:
    raise ValueError("No languages found. Pass --languages or configure multilingual_experiments.languages.")

  results_root = cli_args.results_root or exp_cfg.get('results_root') or 'experiments'
  configs_root = cli_args.configs_root or exp_cfg.get('configs_root')
  write_configs = cli_args.write_configs or exp_cfg.get('write_configs', False)
  if cli_args.no_write_configs:
    write_configs = False

  model_tag = _slug_model_tag(base_args, exp_cfg)
  layer = cli_args.layer
  if layer is None:
    layer = base_args.get('model', {}).get('model_layer')
  if layer is None:
    raise ValueError("No model layer set; add model.model_layer or pass --layer.")

  runs = []
  for experiment_type in experiment_types:
    for eval_lang, train_keys in _build_runs(experiment_type, languages):
      train_spec = _format_train_spec(train_keys, languages, eval_lang)
      runs.append((experiment_type, eval_lang, train_keys, train_spec))

  if cli_args.dry_run:
    for experiment_type, eval_lang, train_keys, train_spec in runs:
      print(f"{experiment_type} eval={eval_lang} train={train_keys} -> {train_spec}")
    return

  for experiment_type, eval_lang, train_keys, train_spec in runs:
    run_args = copy.deepcopy(base_args)
    run_args.setdefault('dataset', {}).setdefault('keys', {})
    run_args['dataset']['keys']['train'] = list(train_keys)
    run_args['dataset']['keys']['dev'] = [eval_lang]
    run_args['dataset']['keys']['test'] = [eval_lang]

    run_args.setdefault('model', {})['model_layer'] = int(layer)
    if 'decoder_model' in run_args:
      run_args['decoder_model']['layer'] = int(layer)

    results_dir = os.path.join(
        results_root,
        _safe_component(model_tag),
        _safe_component(eval_lang),
        _safe_component(experiment_type),
        _safe_component(train_spec),
        f"layer-{int(layer)}",
    )
    os.makedirs(results_dir, exist_ok=True)
    run_args.setdefault('reporting', {})['root'] = results_dir

    config_path = cli_args.config
    if write_configs and configs_root:
      config_path = os.path.join(
          configs_root,
          _safe_component(eval_lang),
          _safe_component(experiment_type),
          _safe_component(train_spec),
          f"layer-{int(layer)}.yaml",
      )
      _write_yaml(config_path, run_args)

    _run_experiment(config_path, results_dir, cli_args.train_probe, cli_args.report_results)


if __name__ == '__main__':
  main()
