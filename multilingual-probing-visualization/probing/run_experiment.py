"""Loads configuration yaml and runs an experiment."""
from argparse import ArgumentParser
import copy
import itertools
import os
from datetime import datetime
import shutil
import yaml
from tqdm import tqdm
import torch
import numpy as np
import random

import data
import model
import probe
import regimen
import reporter
import task
import loss


def dedupe_layers(layers):
  ordered = []
  seen = set()
  for layer in layers:
    layer = int(layer)
    if layer in seen:
      continue
    seen.add(layer)
    ordered.append(layer)
  return ordered


def parse_layers_argument(layers_arg, default_layer):
  """Parses CLI layer selections.

  Supports:
    - Comma lists: "0,5,10,15"
    - Inclusive ranges: "0:20" (step=1)
    - Inclusive stepped ranges: "0:20:5"
    - Mixtures: "0,4:12:4,15"
  """
  if not layers_arg:
    if default_layer is None:
      raise ValueError("No layers provided and no default layer found in config.")
    return [int(default_layer)]

  layers = []
  for raw_item in str(layers_arg).split(','):
    item = raw_item.strip()
    if not item:
      continue
    if ':' in item:
      parts = [x.strip() for x in item.split(':')]
      if len(parts) not in (2, 3):
        raise ValueError("Layer range '{}' must be start:end or start:end:step".format(item))
      start = int(parts[0])
      end = int(parts[1])
      step = int(parts[2]) if len(parts) == 3 else (1 if end >= start else -1)
      if step == 0:
        raise ValueError("Layer step cannot be zero in '{}'".format(item))
      if (end - start) * step < 0:
        raise ValueError("Layer range '{}' has incompatible step direction".format(item))
      stop = end + (1 if step > 0 else -1)
      layers.extend(list(range(start, stop, step)))
    else:
      layers.append(int(item))

  if not layers:
    raise ValueError("No valid layers parsed from '{}'".format(layers_arg))
  return dedupe_layers(layers)


def parse_layers_from_config(config_layers, default_layer):
  """Parses `model.layers` from YAML config.

  Accepted values:
    - int: 5
    - string: "0,5,10" or "0:20:5"
    - list: [0, 5, 10] or ["0:20:5", 23]
  """
  if config_layers is None:
    if default_layer is None:
      raise ValueError("Config must define either model.model_layer or model.layers.")
    return [int(default_layer)]

  if isinstance(config_layers, int):
    return [int(config_layers)]

  if isinstance(config_layers, str):
    return parse_layers_argument(config_layers, default_layer=None)

  if isinstance(config_layers, (list, tuple)):
    layers = []
    for item in config_layers:
      if isinstance(item, int):
        layers.append(item)
      elif isinstance(item, str):
        layers.extend(parse_layers_argument(item, default_layer=None))
      else:
        raise ValueError("Unsupported item in model.layers: {} ({})".format(item, type(item)))
    if not layers:
      raise ValueError("model.layers was provided but no valid layer values were parsed.")
    return dedupe_layers(layers)

  raise ValueError("Unsupported type for model.layers: {}".format(type(config_layers)))


def resolve_layers_to_run(cli_layers_arg, yaml_args):
  default_layer = yaml_args.get('model', {}).get('model_layer')
  config_layers = yaml_args.get('model', {}).get('layers')
  if cli_layers_arg:
    return parse_layers_argument(cli_layers_arg, default_layer), "cli --layers"
  if config_layers is not None:
    return parse_layers_from_config(config_layers, default_layer), "config model.layers"
  if default_layer is None:
    raise ValueError("Config must define model.model_layer (or model.layers or --layers).")
  return [int(default_layer)], "config model.model_layer"


def set_config_layer(yaml_args, layer):
  yaml_args['model']['model_layer'] = int(layer)
  if 'decoder_model' in yaml_args:
    yaml_args['decoder_model']['layer'] = int(layer)


def selector_sweep_keys():
  return {
    'dataset_name': 'dataset_names',
    'variation': 'variations',
    'sentence_type': 'sentence_types',
    'attractor_count': 'attractor_counts',
  }


def _as_list(value):
  if isinstance(value, (list, tuple)):
    return list(value)
  return [value]


def resolve_selector_combinations(yaml_args):
  dataset_cfg = yaml_args.get('dataset', {})
  selector = copy.deepcopy(dataset_cfg.get('selector', {}))
  if not selector:
    return [selector], "default selector"

  key_map = selector_sweep_keys()
  values = {}
  has_sweep = False
  for singular_key, plural_key in key_map.items():
    raw_value = None
    if plural_key in selector:
      raw_value = selector.get(plural_key)
      has_sweep = True
    elif singular_key in selector:
      raw_value = selector.get(singular_key)

    if raw_value is None:
      continue

    parsed = _as_list(raw_value)
    if not parsed:
      raise ValueError(f"dataset.selector.{singular_key}/{plural_key} cannot be empty.")
    values[singular_key] = parsed

  if not has_sweep:
    return [selector], "config dataset.selector"

  missing = [k for k in key_map.keys() if k not in values]
  if missing:
    raise ValueError(
      "Selector sweep requires values for {}. "
      "Set these as singular values or plural lists in dataset.selector.".format(", ".join(missing))
    )

  combinations = []
  for dataset_name, variation, sentence_type, attractor_count in itertools.product(
      values['dataset_name'],
      values['variation'],
      values['sentence_type'],
      values['attractor_count']):
    combo_selector = copy.deepcopy(selector)
    for plural_key in key_map.values():
      combo_selector.pop(plural_key, None)
    combo_selector['dataset_name'] = dataset_name
    combo_selector['variation'] = variation
    combo_selector['sentence_type'] = sentence_type
    combo_selector['attractor_count'] = attractor_count
    combinations.append(combo_selector)

  return combinations, "config dataset.selector sweep"


def apply_selector_to_config(yaml_args, selector):
  dataset_cfg = yaml_args.setdefault('dataset', {})
  selector_cfg = copy.deepcopy(selector)
  for plural_key in selector_sweep_keys().values():
    selector_cfg.pop(plural_key, None)
  dataset_cfg['selector'] = selector_cfg

  corpus_cfg = dataset_cfg.setdefault('corpus', {})
  root_template = corpus_cfg.get('root_template')
  if root_template:
    required = ('dataset_name', 'variation', 'sentence_type', 'attractor_count')
    missing = [k for k in required if k not in selector_cfg]
    if missing:
      raise ValueError(
        "dataset.corpus.root_template requires selector keys: {}.".format(", ".join(missing))
      )
    corpus_cfg['root'] = str(root_template).format(
      dataset_name=selector_cfg['dataset_name'],
      variation=selector_cfg['variation'],
      sentence_type=selector_cfg['sentence_type'],
      attractor_count=selector_cfg['attractor_count'],
    )

  return apply_attractor_batch_scaling(yaml_args, selector_cfg)


def apply_attractor_batch_scaling(yaml_args, selector):
  """Scales batch sizes by attractor count.

  Rule:
    attractor_count=1 -> base batch size
    attractor_count=2 -> base/2
    attractor_count=3 -> base/4
    ...
  """
  if 'attractor_count' not in selector:
    return None

  try:
    attractor_count = int(selector['attractor_count'])
  except (TypeError, ValueError) as exc:
    raise ValueError(f"Invalid attractor_count value: {selector['attractor_count']}") from exc

  if attractor_count < 1:
    raise ValueError(f"attractor_count must be >= 1, got {attractor_count}")

  divisor = 2 ** (attractor_count - 1)
  scaling_info = {
    'attractor_count': attractor_count,
    'divisor': divisor,
  }

  dataset_cfg = yaml_args.get('dataset', {})
  if 'batch_size' in dataset_cfg:
    base_dataset_bs = int(dataset_cfg['batch_size'])
    scaled_dataset_bs = max(1, base_dataset_bs // divisor)
    dataset_cfg['batch_size'] = scaled_dataset_bs
    scaling_info['dataset_batch_size'] = (base_dataset_bs, scaled_dataset_bs)

  decoder_cfg = yaml_args.get('decoder_model', {})
  if 'batch_size' in decoder_cfg:
    base_decoder_bs = int(decoder_cfg['batch_size'])
    scaled_decoder_bs = max(1, base_decoder_bs // divisor)
    decoder_cfg['batch_size'] = scaled_decoder_bs
    scaling_info['decoder_batch_size'] = (base_decoder_bs, scaled_decoder_bs)

  return scaling_info


def selector_result_suffix(selector):
  if not selector:
    return ''
  dataset_name = selector.get('dataset_name')
  variation = selector.get('variation')
  sentence_type = selector.get('sentence_type')
  attractor_count = selector.get('attractor_count')
  if None in (dataset_name, variation, sentence_type, attractor_count):
    return ''
  return os.path.join(
    str(dataset_name),
    str(variation),
    str(sentence_type),
    f"attractors_{attractor_count}",
  )


def model_tag(yaml_args):
  metadata = yaml_args.get('metadata', {})
  if metadata.get('model_tag'):
    return str(metadata['model_tag'])
  decoder_cfg = yaml_args.get('decoder_model', {})
  model_name = decoder_cfg.get('model_name') or yaml_args.get('model', {}).get('model_name')
  if not model_name:
    return 'model'
  return str(model_name).replace('/', '-')


def build_structured_results_dir(yaml_args, date_suffix):
  """Builds corpus-like results paths for structured experiments."""
  reporting_cfg = yaml_args.get('reporting', {})
  layout = reporting_cfg.get('layout')
  if layout != 'corpus':
    return None

  selector = yaml_args.get('dataset', {}).get('selector', {})
  dataset_name = selector.get('dataset_name')
  variation = selector.get('variation')
  sentence_type = selector.get('sentence_type')
  attractor_count = selector.get('attractor_count')
  if None in (dataset_name, variation, sentence_type, attractor_count):
    raise ValueError(
      "reporting.layout='corpus' requires dataset.selector fields: "
      "dataset_name, variation, sentence_type, attractor_count"
    )

  root = reporting_cfg.get('root', '')
  task_name = yaml_args.get('probe', {}).get('task_name', 'task')
  layer = yaml_args.get('model', {}).get('model_layer')
  if layer is None:
    raise ValueError("model.model_layer is required for corpus layout results.")

  return os.path.join(
    root,
    str(dataset_name),
    str(variation),
    str(sentence_type),
    f"attractors_{attractor_count}",
    model_tag(yaml_args),
    f"layer-{layer}",
    str(task_name),
    date_suffix,
  )

def is_polar_probe(args):
  return args.get('probe', {}).get('use_polar', False)

def choose_task_classes(args):
  """Chooses which task class to use based on config.

  Args:
    args: the global config dictionary built by yaml.
  Returns:
    A class to be instantiated as a task specification.
  """
  if args['probe']['task_name'] == 'parse-distance':
    task_class = task.ParseDistanceTask
    if is_polar_probe(args):
      reporter_class = reporter.PolarProbeReporter
      loss_class = loss.PolarProbeLoss
      reporting_methods = args['reporting'].get('reporting_methods', [])
      if 'polar_metrics' not in reporting_methods:
        reporting_methods.append('polar_metrics')
      args['reporting']['reporting_methods'] = reporting_methods
    else:
      reporter_class = reporter.WordPairReporter
      reporting_methods = [m for m in args['reporting'].get('reporting_methods', []) if m != 'polar_metrics']
      args['reporting']['reporting_methods'] = reporting_methods
      if args['probe_training']['loss'] == 'L1':
        loss_class = loss.L1DistanceLoss
      else:
        raise ValueError("Unknown loss type for given probe type: {}".format(
          args['probe_training']['loss']))
  elif args['probe']['task_name'] == 'parse-depth':
    task_class = task.ParseDepthTask
    reporter_class = reporter.WordReporter
    if args['probe_training']['loss'] == 'L1':
      loss_class = loss.L1DepthLoss
    else:
      raise ValueError("Unknown loss type for given probe type: {}".format(
        args['probe_training']['loss']))
  elif args['probe']['task_name'] == 'semantic-roles':
    task_class = task.SemanticRolesTask
    reporter_class = reporter.WordReporter
    if args['probe_training']['loss'] == 'CrossEntropy':
      loss_class = loss.CrossEntropyLoss
    else:
      raise ValueError("Unknown loss type for given probe type: {}".format(
        args['probe_training']['loss']))
  else:
    raise ValueError("Unknown probing task type: {}".format(
      args['probe']['task_name']))
  return task_class, reporter_class, loss_class

def choose_dataset_class(args):
  """Chooses which dataset class to use based on config.

  Args:
    args: the global config dictionary built by yaml.
  Returns:
    A class to be instantiated as a dataset.
  """
  if args['model']['model_type'] in {'ELMo-disk', 'ELMo-random-projection', 'ELMo-decay'}:
    dataset_class = data.ELMoDataset
  elif args['model']['model_type'] == 'BERT-disk':
    dataset_class = data.BERTDataset
  elif args['model']['model_type'] == 'decoder-transformer-lens':
    dataset_class = data.TransformerLensDataset
  else:
    raise ValueError("Unknown model type for datasets: {}".format(
      args['model']['model_type']))

  return dataset_class

def choose_probe_class(args):
  """Chooses which probe and reporter classes to use based on config.

  Args:
    args: the global config dictionary built by yaml.
  Returns:
    A probe_class to be instantiated.
  """
  if is_polar_probe(args) and args['probe']['task_signature'] != 'word_pair':
    raise ValueError("Polar probe requires task_signature 'word_pair'")
  if args['probe']['task_signature'] == 'word':
    if args['probe']['psd_parameters']:
      return probe.OneWordPSDProbe
    else:
      return probe.OneWordNonPSDProbe
  elif args['probe']['task_signature'] == 'word_pair':
    if is_polar_probe(args):
      return probe.PolarProbe
    if args['probe']['psd_parameters']:
      return probe.TwoWordPSDProbe
    else:
      return probe.TwoWordNonPSDProbe
  elif args['probe']['task_signature'] == 'word_label':
    if 'probe_spec' not in args['probe'] or args['probe']['probe_spec']['probe_hidden_layers'] == 0:
      return probe.OneWordLinearLabelProbe
    else:
      return probe.OneWordNNLabelProbe
  else:
    raise ValueError("Unknown probe type (probe function signature): {}".format(
      args['probe']['task_signature']))

def choose_model_class(args):
  """Chooses which reporesentation learner class to use based on config.

  Args:
    args: the global config dictionary built by yaml.
  Returns:
    A class to be instantiated as a model to supply word representations.
  """
  if args['model']['model_type'] == 'ELMo-disk':
    return model.DiskModel
  elif args['model']['model_type'] == 'BERT-disk':
    return model.DiskModel
  elif args['model']['model_type'] == 'decoder-transformer-lens':
    return model.DiskModel
  elif args['model']['model_type'] == 'ELMo-random-projection':
    return model.ProjectionModel
  elif args['model']['model_type'] == 'ELMo-decay':
    return model.DecayModel
  elif args['model']['model_type'] == 'pytorch_model':
    raise ValueError("Using pytorch models for embeddings not yet supported...")
  else:
    raise ValueError("Unknown model type: {}".format(
      args['model']['model_type']))

def choose_regimen_class(args):
  if is_polar_probe(args):
    return regimen.PolarProbeRegimen
  return regimen.ProbeRegimen

def run_train_probe(args, probe, dataset, model, loss, reporter, regimen):
  """Trains a structural probe according to args.

  Args:
    args: the global config dictionary built by yaml.
          Describes experiment settings.
    probe: An instance of probe.Probe or subclass.
          Maps hidden states to linguistic quantities.
    dataset: An instance of data.SimpleDataset or subclass.
          Provides access to DataLoaders of corpora.
    model: An instance of model.Model
          Provides word representations.
    reporter: An instance of reporter.Reporter
          Implements evaluation and visualization scripts.
  Returns:
    None; causes probe parameters to be written to disk.
  """
  regimen.train_until_convergence(probe, model, loss,
      dataset.get_train_dataloader(), dataset.get_dev_dataloader())
  args['did_train'] = True


def run_report_results(args, probe, dataset, model, loss, reporter, regimen):
  """
  Reports results from a structural probe according to args.
  By default, does so only for dev set.
  Requires a simple code change to run on the test set.
  """
  probe_params_path = os.path.join(args['reporting']['root'],args['probe']['params_path'])

  dev_dataloader = dataset.get_dev_dataloader()
  try:
    probe.load_state_dict(torch.load(probe_params_path))
    probe.eval()
    dev_predictions = regimen.predict(probe, model, dev_dataloader)
  except FileNotFoundError:
    print("No trained probe found.")
    dev_predictions = None

  tqdm.write('[Step] Generating dev-set reports...')
  reporter(dev_predictions, probe, model, dev_dataloader, 'dev')

  #train_dataloader = dataset.get_train_dataloader(shuffle=False)
  #train_predictions = regimen.predict(probe, model, train_dataloader)
  #reporter(train_predictions, train_dataloader, 'train')

  test_dataloader = dataset.get_test_dataloader()
  try:
    test_predictions = regimen.predict(probe, model, test_dataloader)
    reporter(test_predictions, probe, model, test_dataloader, 'test')
  except Exception as e:
    tqdm.write(f"Error running on test set: {e}")

def execute_experiment(args, train_probe, report_results):
  """
  Execute an experiment as determined by the configuration
  in args.

  Args:
    train_probe: Boolean whether to train the probe
    report_results: Boolean whether to report results
  """
  dataset_class = choose_dataset_class(args)
  task_class, reporter_class, loss_class = choose_task_classes(args)
  probe_class = choose_probe_class(args)
  model_class = choose_model_class(args)
  regimen_class = choose_regimen_class(args)

  task = task_class()
  expt_dataset = dataset_class(args, task)
  if hasattr(expt_dataset, 'train_dataset'):
    train_count = len(expt_dataset.train_dataset) if args['train_probe'] else 0
    dev_count = len(expt_dataset.dev_dataset)
    tqdm.write('[Step] Dataset ready (train: {}, dev: {})'.format(train_count, dev_count))
  expt_reporter = reporter_class(args)
  if hasattr(expt_reporter, 'set_dataset'):
    expt_reporter.set_dataset(expt_dataset)
  expt_probe = probe_class(args)
  expt_model = model_class(args)
  expt_regimen = regimen_class(args)
  expt_loss = loss_class(args)

  if train_probe:
    print('Training probe...')
    run_train_probe(args, expt_probe, expt_dataset, expt_model, expt_loss, expt_reporter, expt_regimen)
    tqdm.write('[Step] Training complete, saved probe parameters.')
  if report_results:
    print('Reporting results of trained probe...')
    run_report_results(args, expt_probe, expt_dataset, expt_model, expt_loss, expt_reporter, expt_regimen)


def setup_new_experiment_dir(args, yaml_args, reuse_results_path):
  """Constructs a directory in which results and params will be stored.

  If reuse_results_path is not None, then it is reused; no new
  directory is constrcted.

  Args:
    args: the command-line arguments:
    yaml_args: the global config dictionary loaded from yaml
    reuse_results_path: the (optional) path to reuse from a previous run.
  """
  now = datetime.now()
  date_suffix = '-'.join((str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond]))
  model_layer = str(yaml_args['model']['model_layer'])
  model_suffix = '-'.join(("model-layer", model_layer, yaml_args['probe']['task_name']))
  if reuse_results_path:
    new_root = reuse_results_path
    tqdm.write('Reusing old results directory at {}'.format(new_root))
    if args.train_probe == -1:
      args.train_probe = 0
      tqdm.write('Setting train_probe to 0 to avoid squashing old params; '
          'explicitly set to 1 to override.')
  else:
    new_root = build_structured_results_dir(yaml_args, date_suffix)
    if new_root is None:
      new_root = os.path.join(yaml_args['reporting']['root'], model_suffix + '-' + date_suffix +'/' )
    tqdm.write('Constructing new results directory at {}'.format(new_root))
  yaml_args['reporting']['root'] = new_root
  os.makedirs(new_root, exist_ok=True)
  try:
    shutil.copyfile(args.experiment_config, os.path.join(yaml_args['reporting']['root'],
      os.path.basename(args.experiment_config)))
  except shutil.SameFileError:
    tqdm.write('Note, the config being used is the same as that already present in the results dir')


if __name__ == '__main__':
  argp = ArgumentParser()
  argp.add_argument('experiment_config')
  argp.add_argument('--results-dir', default='',
      help='Set to reuse an old results dir; '
      'if left empty, new directory is created')
  argp.add_argument('--layers', default='',
      help='Comma list/ranges of layers, e.g. "0,5,10,15" or "0:20:5"')
  argp.add_argument('--train-probe', default=-1, type=int,
      help='Set to train a new probe.; ')
  argp.add_argument('--report-results', default=1, type=int,
      help='Set to report results; '
      '(optionally after training a new probe)')
  argp.add_argument('--seed', default=None, type=int,
      help='sets all random seeds for (within-machine) reproducibility')
  cli_args = argp.parse_args()
  with open(cli_args.experiment_config, 'r') as config_file:
    base_yaml_args = yaml.safe_load(config_file)
  if cli_args.seed is None:
    cli_args.seed = base_yaml_args.get('seed', 42)
  base_yaml_args['seed'] = cli_args.seed
  print(f"Seed: {cli_args.seed}")
  if cli_args.seed is not None:
    random.seed(cli_args.seed)
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    torch.cuda.manual_seed_all(cli_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

  layers, layer_source = resolve_layers_to_run(cli_args.layers, base_yaml_args)
  print("Layers to run (from {}): {}".format(layer_source, layers))
  selectors, selector_source = resolve_selector_combinations(base_yaml_args)
  print("Selector combinations (from {}): {}".format(selector_source, len(selectors)))

  for selector in selectors:
    selector_info = selector_result_suffix(selector)
    if selector_info:
      tqdm.write("=" * 80)
      tqdm.write(f"[Step] Starting selector {selector_info}")
    for layer in layers:
      tqdm.write("-" * 80)
      tqdm.write(f"[Step] Starting layer {layer}")
      yaml_args = copy.deepcopy(base_yaml_args)
      batch_scaling_info = apply_selector_to_config(yaml_args, selector)
      set_config_layer(yaml_args, layer)

      if batch_scaling_info:
        ds_bs = batch_scaling_info.get('dataset_batch_size')
        dec_bs = batch_scaling_info.get('decoder_batch_size')
        details = [f"attractor={batch_scaling_info['attractor_count']} (x1/{batch_scaling_info['divisor']})"]
        if ds_bs:
          details.append(f"dataset.batch_size {ds_bs[0]}->{ds_bs[1]}")
        if dec_bs:
          details.append(f"decoder_model.batch_size {dec_bs[0]}->{dec_bs[1]}")
        tqdm.write("[Step] Batch scaling: " + ", ".join(details))

      reuse_results_dir = cli_args.results_dir
      if reuse_results_dir:
        if len(selectors) > 1 and selector_info:
          reuse_results_dir = os.path.join(reuse_results_dir, selector_info)
        if len(layers) > 1:
          reuse_results_dir = os.path.join(reuse_results_dir, f"layer-{layer}")

      setup_new_experiment_dir(cli_args, yaml_args, reuse_results_dir)
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      yaml_args['device'] = device
      yaml_args['train_probe'] = cli_args.train_probe
      yaml_args['did_train'] = (
        os.path.exists(os.path.join(yaml_args['reporting']['root'], yaml_args['probe']['params_path']))
      )
      execute_experiment(yaml_args, train_probe=cli_args.train_probe, report_results=cli_args.report_results)
