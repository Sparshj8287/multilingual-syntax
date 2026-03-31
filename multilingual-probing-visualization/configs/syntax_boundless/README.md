Syntax Boundless configs for `probing/run_experiment.py`.

Files:
- `gemma-3-12b-pt.yaml`
- `qwen-3-8b.yaml`
- `olmo-3-7b.yaml`
- `llama-3.1-8b.yaml`

Each config includes:
- `dataset.selector.dataset_name`
- `dataset.selector.variation`
- `dataset.selector.sentence_type` (`base_sentence` or `source_sentence`)
- `dataset.selector.attractor_count` (1..6)

Important:
- `run_experiment.py` does not automatically expand selector fields.
- Keep `dataset.corpus.root` consistent with selector values.
- Ready splits are under `datasets/syntax_boundless_das_ready/...`.

Example run:
`python probing/run_experiment.py configs/syntax_boundless/llama-3.1-8b.yaml --train-probe 1 --report-results 1`

Multi-layer run examples:
- `python probing/run_experiment.py configs/syntax_boundless/llama-3.1-8b.yaml --layers 0,5,10,15 --train-probe 1 --report-results 1`
- `python probing/run_experiment.py configs/syntax_boundless/llama-3.1-8b.yaml --layers 0:40:5 --train-probe 1 --report-results 1`

Config-driven layers (no `--layers` needed):
- Add `model.layers` in yaml, for example:
  - `layers: [0, 5, 10, 15]`
  - or `layers: "0:40:5"`
- CLI `--layers` overrides `model.layers` if both are provided.

