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
- Results are saved using corpus-style layout when `reporting.layout: corpus`:
  - `experiments/syntax_boundless/<dataset_name>/<variation>/<sentence_type>/attractors_<n>/<model_tag>/layer-<L>/<task>/<timestamp>/`

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

Config-driven selector sweeps (single command over multiple dataset conditions):
- Add list fields under `dataset.selector`:
  - `dataset_names: [obj_rel_across_anim, obj_rel_across_anim_2]`
  - `variations: [linear, bow]`
  - `attractor_counts: [1, 2, 3, 4]`
- Keep singular defaults (`dataset_name`, `variation`, `sentence_type`, `attractor_count`) too.
- Set `dataset.corpus.root_template` so each run resolves the right corpus root:
  - `datasets/syntax_boundless_das_ready/{dataset_name}/{variation}/{sentence_type}/attractors_{attractor_count}`

