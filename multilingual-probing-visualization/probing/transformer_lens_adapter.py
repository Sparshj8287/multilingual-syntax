"""Utilities for extracting decoder-only activations via TransformerLens."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from tqdm import tqdm



def _maybe_use_vendor_transformer_lens(enabled: bool = True):
  """Prefer the repo-local TransformerLens checkout when present."""

  if not enabled:
    return None

  repo_root = Path(__file__).resolve().parents[1]
  vendor_root = (repo_root / 'TransformerLens').resolve()
  if not vendor_root.exists():
    return None

  vendor_path = str(vendor_root)
  if vendor_path not in sys.path:
    sys.path.insert(0, vendor_path)

  existing_module = sys.modules.get('transformer_lens')
  if existing_module is not None:
    module_file = getattr(existing_module, '__file__', None)
    reload_needed = True
    if module_file:
      try:
        module_path = Path(module_file).resolve()
        reload_needed = not module_path.is_relative_to(vendor_root)
      except Exception:
        reload_needed = True
    if reload_needed:
      for name in list(sys.modules.keys()):
        if name == 'transformer_lens' or name.startswith('transformer_lens.'):
          sys.modules.pop(name, None)

  return vendor_root


class TransformerLensEmbeddingExtractor:
  """Materializes word-level representations from decoder-only models.

  Uses TransformerLens to run a HookedTransformer and average subword activations
  so the rest of the probing pipeline can keep assuming word-aligned vectors.
  """

  def __init__(self, args):
    decoder_cfg = args.get('decoder_model', {})
    if not decoder_cfg:
      raise ValueError("Expected a `decoder_model` section in the experiment config.")

    self._configure_proxy(decoder_cfg)

    use_vendor = decoder_cfg.get('use_vendor_transformer_lens', True)
    env_override = os.environ.get('USE_VENDOR_TRANSFORMER_LENS')
    if env_override is not None:
      use_vendor = env_override.strip().lower() not in {'0', 'false', 'no', 'off'}
    self._transformer_lens_root = _maybe_use_vendor_transformer_lens(enabled=use_vendor)
    try:
      from transformer_lens import HookedTransformer
    except ImportError as exc:
      raise ImportError(
          "TransformerLensEmbeddingExtractor requires the transformer_lens package. "
          "Install it via `pip install transformer_lens`."
      ) from exc
    if self._transformer_lens_root:
      tqdm.write(f"[TLens] Using TransformerLens from {self._transformer_lens_root}")
    elif not use_vendor:
      tqdm.write("[TLens] Using system transformer_lens (vendored copy disabled).")
    try:
      from transformers import AutoTokenizer
    except ImportError as exc:
      raise ImportError(
          "TransformerLensEmbeddingExtractor requires the transformers package."
      ) from exc

    self.device = args.get('device', torch.device('cpu'))
    self.activation_name = decoder_cfg.get('activation_name', 'resid_post')
    self.layer_index = self._resolve_layer_index(decoder_cfg.get('layer', args['model'].get('model_layer')))
    self.batch_size = max(1, int(decoder_cfg.get('batch_size', 1)))
    self.cache_dir = self._normalize_path(decoder_cfg.get('cache_dir'))
    self.local_model_path = self._normalize_path(decoder_cfg.get('local_path'))
    if self.local_model_path and not os.path.exists(self.local_model_path):
      raise FileNotFoundError(
          f"decoder_model.local_path='{self.local_model_path}' does not exist. "
          "Set it to 'none' to download from HuggingFace instead.")

    dtype = self._parse_dtype(decoder_cfg.get('dtype', 'float16'))
    model_name = decoder_cfg.get('model_name')
    if not model_name:
      raise ValueError("decoder_model.model_name is required (even when using a local_path).")
    model_source = model_name
    model_kwargs = {
        'device': str(self.device),
        'dtype': dtype,
        'fold_ln': decoder_cfg.get('fold_ln', False),
        'center_writing_weights': decoder_cfg.get('center_writing_weights', False),
    }
    self.model = HookedTransformer.from_pretrained(model_source, **model_kwargs)
    self.model.eval()
    self.hidden_size = self.model.cfg.d_model
    self.total_layers = self.model.cfg.n_layers
    self.layer_index = self._normalize_layer_index(self.layer_index, self.total_layers)

    tokenizer_kwargs = {'use_fast': True, 'padding_side': decoder_cfg.get('padding_side', 'right')}
    tokenizer_source = model_name
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.bos_token
    if self.tokenizer.pad_token is None:
      raise ValueError("Tokenizer must expose either a PAD, EOS, or BOS token for padding.")

  @staticmethod
  def _parse_dtype(dtype_str: str):
    normalized = dtype_str.lower()
    if normalized in {'fp16', 'float16', 'half'}:
      return torch.float16
    if normalized in {'bf16', 'bfloat16'}:
      return torch.bfloat16
    if normalized in {'fp32', 'float32'}:
      return torch.float32
    raise ValueError(f"Unsupported dtype '{dtype_str}'. Use float16, bfloat16, or float32.")

  @staticmethod
  def _resolve_layer_index(layer_value):
    if layer_value is None:
      raise ValueError("Set `model.model_layer` or `decoder_model.layer` to choose an activation depth.")
    return int(layer_value)

  @staticmethod
  def _normalize_layer_index(index: int, total_layers: int):
    if index < 0:
      index = total_layers + index
    if index < 0 or index >= total_layers:
      raise ValueError(f"Layer index {index} out of bounds for model with {total_layers} layers.")
    return index

  @staticmethod
  def _normalize_path(path_value):
    if path_value is None:
      return None
    if isinstance(path_value, str):
      stripped = path_value.strip()
      if stripped == '' or stripped.lower() == 'none':
        return None


      return os.path.expanduser(stripped)
    return path_value

  @staticmethod
  def _configure_proxy(decoder_cfg):
    proxy_url = decoder_cfg.get('proxy_url', 'http://xen03.iitd.ac.in:3128')
    if not proxy_url:
      return
    for key in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY'):
      if key not in os.environ:
        os.environ[key] = proxy_url

  def encode_observations(self, observations: Sequence):
    """Returns word-level tensors for every observation."""
    outputs: List[torch.Tensor] = []
    total_batches = max(1, (len(observations) + self.batch_size - 1) // self.batch_size)
    for batch in tqdm(self._batch_iter(observations, self.batch_size),
                      total=total_batches,
                      desc='[TLens] encoding batches',
                      leave=False):
      outputs.extend(self._encode_batch(batch))
    return outputs

  def _encode_batch(self, batch):
    sentences = [self._observation_to_sentence(obs) for obs in batch]
    encoded = self.tokenizer(
        sentences,
        return_tensors='pt',
        padding=True,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop('offset_mapping')
    tokens = encoded['input_ids'].to(self.device)
    with torch.no_grad():
      _, cache = self.model.run_with_cache(
          tokens,
          names_filter=lambda name: self.activation_name in name,
      )
      activation = cache[self.activation_name, self.layer_index].detach().cpu()
    batch_word_embeddings = []
    for obs, token_vectors, offset in zip(batch, activation, offsets):
      word_embeddings = self._aggregate_to_words(token_vectors.float(), offset.tolist(), obs.sentence)
      batch_word_embeddings.append(word_embeddings)
    del cache
    return batch_word_embeddings

  @staticmethod
  def _observation_to_sentence(observation):
    sentence = observation.sentence
    if isinstance(sentence, (list, tuple)):
      return ' '.join(sentence)
    return str(sentence)

  @staticmethod
  def _word_spans(words: Sequence[str]):
    spans = []
    cursor = 0
    for word in words:
      word_str = str(word)
      spans.append((cursor, cursor + len(word_str)))
      cursor += len(word_str) + 1
    return spans

  def _aggregate_to_words(self, token_vectors: torch.Tensor, offsets, words):
    word_spans = self._word_spans(words)
    alignment: List[List[int]] = [[] for _ in word_spans]
    for token_index, (start, end) in enumerate(offsets):
      if start == end == 0:
        continue  # skip special tokens
      word_index = self._first_intersecting_span(word_spans, start, end)
      if word_index is not None:
        alignment[word_index].append(token_index)
    aggregated = []
    for idx, token_indices in enumerate(alignment):
      if not token_indices:
        nearest = self._nearest_token(offsets, word_spans[idx])
        if nearest is not None:
          token_indices = [nearest]
      if not token_indices:
        raise ValueError(
            f"Unable to align word '{words[idx]}' (index {idx}) "
            "with any tokenizer pieces. Inspect your tokenization settings."
        )
      aggregated.append(token_vectors[token_indices].mean(dim=0))
    return torch.stack(aggregated)

  @staticmethod
  def _first_intersecting_span(spans, start, end):
    for idx, (word_start, word_end) in enumerate(spans):
      if start < word_end and end > word_start:
        return idx
    return None

  @staticmethod
  def _nearest_token(offsets, word_span):
    valid_indices = [idx for idx, (s, e) in enumerate(offsets) if not (s == e == 0)]
    if not valid_indices:
      return None
    word_center = sum(word_span) / 2.0
    best_index = min(
        valid_indices,
        key=lambda idx: abs((offsets[idx][0] + offsets[idx][1]) / 2.0 - word_center),
    )
    return best_index

  @staticmethod
  def _batch_iter(seq: Sequence, batch_size: int) -> Iterable[List]:
    for start in range(0, len(seq), batch_size):
      yield list(seq[start:start + batch_size])
