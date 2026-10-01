# MAR Metric

Compute the Matrix Alignment Ratio (MAR) between learned Boundless DAS subspaces.

The script uses:

- Results: `../results_das_full_with_saved_boundaries`
- Checkpoints: read from each `result.json` under `training.best_rotation_checkpoint.path`
- Basis extraction: `W = R[:, :ceil(b2)]`
- MAR:

```python
overlap = W_a.T @ W_b
mar = ||overlap||_F^2 / sqrt(m_a * m_b)
```

Run from this folder, using the same Python environment used for BDAS training:

```bash
python compute_mar.py
```

By default, the script includes Gemma, Qwen, OLMo, and Llama. You can also
select models with aliases or exact result directory names:

```bash
python compute_mar.py --models llama olmo
python compute_mar.py --models llama-3.1-8b olmo-3-7b
python compute_mar.py --models all
```

By default this includes both setups:

- `source_to_base`: SG -> PL
- `base_to_source`: PL -> SG

Outputs are written to `outputs/`:

- `mar_results.csv`: syntax-vs-proximity and syntax-vs-majority MAR values
- `boundary_m_by_hypothesis.csv`: learned plot size `m = floor(b2)` per hypothesis
- `mar_summary.json`: combined machine-readable summary
- `boundary_m_by_model_hypothesis.png`: grouped bar plot of `m` by model, setup, and hypothesis

By default, syntax is variation-matched:

- `syntax_linear_vs_prox`: `obj_rel_across_anim/linear` vs `obj_rel_across_anim_2/linear`
- `syntax_bow_vs_maj`: `obj_rel_across_anim/bow` vs `obj_rel_across_anim_2/bow`

Layer numbers are not used for matching. For each model, setup, and hypothesis source,
the script uses the available trained layer and records those layer numbers in the CSVs.

For the bar plot, `H_syn` uses the mean `m = floor(b2)` across syntax linear and syntax bow runs.
MAR basis extraction still uses `ceil(b2)`, matching `W = R[:, :ceil(b2)]`.
