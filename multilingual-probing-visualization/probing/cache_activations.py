"""Precompute and cache TransformerLens activations for datasets."""

from argparse import ArgumentParser
import os
from pathlib import Path
import yaml
import torch
from tqdm import tqdm

import data
import transformer_lens_adapter


def _slug_model_tag(args):
  if 'metadata' in args and args['metadata'].get('model_tag'):
    return str(args['metadata']['model_tag'])
  decoder_cfg = args.get('decoder_model', {})
  model_name = decoder_cfg.get('model_name') or args.get('model', {}).get('model_name')
  if not model_name:
    return 'model'
  return str(model_name).replace('/', '-')


def _format_path(path_value, replacements):
  if path_value is None:
    return None
  path_value = str(path_value)
  for token, value in replacements.items():
    if value is not None:
      path_value = path_value.replace(f'{{{token}}}', str(value))
  return path_value


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


def _resolve_cache_path(embeddings_cfg, replacements, split, lang):
  root_template = embeddings_cfg.get('root')
  path_template = embeddings_cfg.get(f'{split}_path')
  if not path_template:
    path_template = os.path.join('{lang}', f'{split}.pt')
  root = _format_path(root_template, replacements)
  path = _format_path(path_template, replacements)
  path = str(path).replace('{lang}', lang)
  return os.path.join(root, path) if root else path


def _parse_layers(args, cli_args):
  if cli_args.layers:
    return [int(x) for x in cli_args.layers.split(',') if x.strip()]
  if cli_args.start_layer is not None and cli_args.end_layer is not None:
    return list(range(cli_args.start_layer, cli_args.end_layer + 1))
  model_layer = args.get('model', {}).get('model_layer')
  if model_layer is None:
    raise ValueError("No layer provided. Set model.model_layer in config or pass --layers/--start-layer.")
  return [int(model_layer)]


def main():
  parser = ArgumentParser(description="Cache TransformerLens activations to disk.")
  parser.add_argument('--config', required=True, help='Path to experiment config yaml')
  parser.add_argument('--languages', default='', help='Comma-separated language codes to cache')
  parser.add_argument('--splits', default='train,dev,test', help='Comma-separated splits to cache')
  parser.add_argument('--layers', default='', help='Comma-separated layer indices to cache')
  parser.add_argument('--start-layer', type=int, default=None, help='Start layer index (inclusive)')
  parser.add_argument('--end-layer', type=int, default=None, help='End layer index (inclusive)')
  parser.add_argument('--overwrite', action='store_true', help='Overwrite existing cache files')
  cli_args = parser.parse_args()

  with open(cli_args.config, 'r') as config_file:
    args = yaml.safe_load(config_file)

  args['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  dataset_cfg = args.get('dataset', {})
  embeddings_cfg = dataset_cfg.get('embeddings', {})
  if not embeddings_cfg:
    embeddings_cfg = {'root': os.path.join('activations', _slug_model_tag(args))}

  split_paths = {
      'train': dataset_cfg.get('corpus', {}).get('train_path', 'train.conllu'),
      'dev': dataset_cfg.get('corpus', {}).get('dev_path', 'dev.conllu'),
      'test': dataset_cfg.get('corpus', {}).get('test_path', 'test.conllu'),
  }
  splits = [s.strip() for s in cli_args.splits.split(',') if s.strip()]

  if cli_args.languages:
    languages = [x.strip() for x in cli_args.languages.split(',') if x.strip()]
  else:
    root_path = dataset_cfg.get('corpus', {}).get('root', 'datasets')
    languages = _discover_languages(root_path, split_paths)
  if not languages:
    raise ValueError("No languages found. Pass --languages or ensure dataset root has language folders.")

  by_key = embeddings_cfg.get('by_key')
  if by_key is None:
    by_key = True
  by_key = bool(by_key)

  layers = _parse_layers(args, cli_args)

  extractor = transformer_lens_adapter.TransformerLensEmbeddingExtractor(args)
  args.setdefault('model', {})['hidden_dim'] = extractor.hidden_size

  for layer in layers:
    args['model']['model_layer'] = int(layer)
    if 'decoder_model' in args:
      args['decoder_model']['layer'] = int(layer)
    extractor.layer_index = extractor._normalize_layer_index(int(layer), extractor.total_layers)

    replacements = {
        'layer': layer,
        'activation': args.get('decoder_model', {}).get('activation_name'),
        'model': _slug_model_tag(args),
    }

    for split in splits:
      if by_key:
        for lang in languages:
          cache_path = _resolve_cache_path(embeddings_cfg, replacements, split, lang)
          if os.path.exists(cache_path) and not cli_args.overwrite:
            tqdm.write(f"[cache] Exists, skipping: {cache_path}")
            continue
          tqdm.write(f"[cache] Loading {split} observations for {lang}...")
          observations = data.load_observations(
              args, split, keys=[lang], skip_lines=(split == 'train'))
          if not observations:
            tqdm.write(f"[cache] No observations for {lang}/{split}, skipping.")
            continue
          tqdm.write(f"[cache] Encoding {len(observations)} sentences at layer {layer}...")
          embeddings = extractor.encode_observations(observations)
          os.makedirs(os.path.dirname(cache_path), exist_ok=True)
          torch.save(embeddings, cache_path)
          tqdm.write(f"[cache] Saved: {cache_path}")
      else:
        cache_path = _resolve_cache_path(embeddings_cfg, replacements, split, 'all')
        if '{lang}' in str(cache_path):
          raise ValueError("Embeddings path template includes {lang} but by_key is False.")
        if os.path.exists(cache_path) and not cli_args.overwrite:
          tqdm.write(f"[cache] Exists, skipping: {cache_path}")
          continue
        tqdm.write(f"[cache] Loading {split} observations for all languages...")
        observations = data.load_observations(
            args, split, keys=languages, skip_lines=(split == 'train'))
        if not observations:
          tqdm.write(f"[cache] No observations for split {split}, skipping.")
          continue
        tqdm.write(f"[cache] Encoding {len(observations)} sentences at layer {layer}...")
        embeddings = extractor.encode_observations(observations)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(embeddings, cache_path)
        tqdm.write(f"[cache] Saved: {cache_path}")


if __name__ == '__main__':
  main()
