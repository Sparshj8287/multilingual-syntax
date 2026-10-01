#!/usr/bin/env python3
"""Layer-matched MAR for the position-resolved DAS runs (reviewer H2Df).

The question is whether the syntax/heuristic subspace overlap reported in the
paper is a property of the mechanism or of the single token that every
intervention edited. This computes MAR separately at each intervention site, so
the overlap can be read as a function of position rather than at one point.

Basis extraction and the MAR formula are imported from compute_mar.py so the
numbers are produced by exactly the same code as the main table.

Pairs compared, at a matched (model, seed, position, direction, layer):
    obj_rel_across_anim/linear  vs  obj_rel_across_anim_2/linear  -> MAR_prox
    obj_rel_across_anim/bow     vs  obj_rel_across_anim_2/bow     -> MAR_maj
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from compute_mar import basis_from_R_and_b2, load_rotation_matrix, mar_from_bases

SCRIPT_DIR = Path(__file__).resolve().parent
BDAS_ROOT = SCRIPT_DIR.parent

POSITION_DIR_RE = re.compile(r"^position_(\w+)$")
SEED_DIR_RE = re.compile(r"^seed_(\d+)$")

PAIRS = {
    "MAR_prox": (("obj_rel_across_anim", "linear"), ("obj_rel_across_anim_2", "linear")),
    "MAR_maj": (("obj_rel_across_anim", "bow"), ("obj_rel_across_anim_2", "bow")),
}


def boundary_from_checkpoint(path: Path) -> tuple[float | None, int]:
    """Recover (b2, embed_dim) for runs whose result.json lacks final.boundaries."""
    if not path.exists():
        return None, 0
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = (checkpoint.get("intervention_state") or [None])[0]
    if not state or "intervention_boundaries" not in state:
        return None, 0
    boundary = state["intervention_boundaries"].detach().flatten()[0]
    rotate_state = state["rotate_layer_state_dict"]
    weight = rotate_state.get("weight")
    if weight is None:
        weight = rotate_state["parametrizations.weight.original"]
    embed_dim = int(weight.shape[0])
    b2 = float(torch.clamp(boundary, 1e-3, 1).item()) * embed_dim
    return b2, embed_dim


def position_of(result_path: Path, root: Path, default: str) -> str:
    for part in result_path.relative_to(root).parts:
        m = POSITION_DIR_RE.match(part)
        if m:
            return m.group(1)
    return default


def index_root(
    root: Path,
    default_position: str,
    nua: int,
    models: set[str] | None = None,
    layers: set[int] | None = None,
) -> dict:
    """Key: (model, seed, position, direction, layer, dataset, variation).

    Model and layer filtering happen before the boundary is reconstructed, because
    that reconstruction has to read the checkpoint from disk and is the dominant
    cost of indexing.
    """
    index: dict[tuple, dict[str, Any]] = {}
    if not root.exists():
        print(f"[warn] results root does not exist: {root}")
        return index

    for result_path in root.rglob("result.json"):
        payload = json.loads(result_path.read_text())
        if int(payload["dataset"]["nua"]) != nua:
            continue
        if models is not None and str(payload["model"]["name"]) not in models:
            continue
        if layers is not None and int(payload["intervention"]["layer"]) not in layers:
            continue

        ckpt = payload.get("training", {}).get("best_rotation_checkpoint", {})
        ckpt_path = ckpt.get("path")
        if not ckpt_path:
            continue

        boundaries = payload.get("final", {}).get("boundaries") or []
        if boundaries:
            b2 = boundaries[0].get("b2", boundaries[0].get("b2_raw"))
            embed_dim = int(boundaries[0]["embed_dim"])
        else:
            # Runs saved before final.boundaries was recorded (e.g. results_das_full_es)
            # still carry the boundary inside the checkpoint. Reconstruct b2 with the
            # same arithmetic as summarize_intervention_boundaries():
            #     b2 = clamp(intervention_boundaries, 1e-3, 1) * embed_dim
            b2, embed_dim = boundary_from_checkpoint(Path(ckpt_path))
            if b2 is None:
                continue

        key = (
            str(payload["model"]["name"]),
            int(payload["training"]["seed"]),
            position_of(result_path, root, default_position),
            str(payload["intervention"]["direction"]),
            int(payload["intervention"]["layer"]),
            str(payload["dataset"]["dataset_name"]),
            str(payload["dataset"]["variation"]),
        )
        index[key] = {
            "b2": float(b2),
            "embed_dim": embed_dim,
            "checkpoint": Path(ckpt_path),
            "iia": float(payload["final"]["test"]["accuracy"]) * 100.0,
        }
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--position-root", type=Path, default=BDAS_ROOT / "results_das_positions"
    )
    parser.add_argument(
        "--last-root",
        type=Path,
        default=BDAS_ROOT / "results_das_full_with_saved_boundaries",
        help="Main-paper runs, treated as the 'last' (pre-verb token) position.",
    )
    parser.add_argument("--nua", type=int, default=4)
    parser.add_argument("--models", nargs="+", default=["gemma-3-12b-pt", "qwen-3-8b"])
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Restrict to these layers (e.g. --layers 0 10 20 30 40). Default: all.",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs_positions")
    args = parser.parse_args()

    keep = set(args.models)
    keep_layers = set(args.layers) if args.layers is not None else None
    index = index_root(args.position_root, "unknown", args.nua, keep, keep_layers)
    index.update(index_root(args.last_root, "last", args.nua, keep, keep_layers))
    print(f"[index] {len(index)} runs")

    cache: dict[Path, torch.Tensor] = {}

    def basis(entry: dict[str, Any]) -> torch.Tensor:
        path = entry["checkpoint"]
        if path not in cache:
            cache[path] = load_rotation_matrix(path, entry["embed_dim"])
        return basis_from_R_and_b2(cache[path], entry["b2"])

    rows: list[dict[str, Any]] = []
    combos = sorted({k[:5] for k in index})
    for model, seed, position, direction, layer in combos:
        for pair_name, ((ds_a, var_a), (ds_b, var_b)) in PAIRS.items():
            key_a = (model, seed, position, direction, layer, ds_a, var_a)
            key_b = (model, seed, position, direction, layer, ds_b, var_b)
            if key_a not in index or key_b not in index:
                continue
            entry_a, entry_b = index[key_a], index[key_b]
            W_a, W_b = basis(entry_a), basis(entry_b)
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "position": position,
                    "direction": direction,
                    "layer": layer,
                    "comparison": pair_name,
                    "mar": round(mar_from_bases(W_a, W_b), 4),
                    "m_syn": W_a.shape[1],
                    "m_heur": W_b.shape[1],
                    "iia_syn": round(entry_a["iia"], 2),
                    "iia_heur": round(entry_b["iia"], 2),
                }
            )
            print(
                f"{model:14s} seed{seed:<5d} {position:5s} {direction:15s} L{layer:<3d} "
                f"{pair_name:9s} MAR={rows[-1]['mar']:.4f}  m={W_a.shape[1]}/{W_b.shape[1]}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "mar_by_position.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[saved] {out_csv}  ({len(rows)} comparisons)")


if __name__ == "__main__":
    main()
