#!/usr/bin/env python3
"""
Monte Carlo random-subspace null distribution for MAR.

By default this script reads outputs/mar_summary.json, extracts each empirical
MAR row, infers the ambient dimension from the referenced result.json files,
and writes null-distribution statistics for every row.

For a one-off calculation, set RUN_MODE = "single" and edit SINGLE_CONFIG.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
from scipy.stats import norm


SCRIPT_DIR = Path(__file__).resolve().parent

# -------------------------
# Easy-to-edit run settings
# -------------------------
RUN_MODE = "summary"  # "summary" to use mar_summary.json, or "single"
SUMMARY_PATH = SCRIPT_DIR / "outputs" / "mar_summary.json"
OUTPUT_DIR = SCRIPT_DIR / "outputs_null"
NUM_SAMPLES = 1000
RANDOM_SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

SINGLE_CONFIG = {
    "d": 4096,
    "m_syn": 228,
    "m_heur": 319,
    "empirical_mar": 0.5930837392807007,
    "label": "manual_config",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random-subspace null distributions for MAR."
    )
    parser.add_argument(
        "--mode",
        choices=["summary", "single"],
        default=RUN_MODE,
        help="Use mar_summary.json rows or the SINGLE_CONFIG values.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=SUMMARY_PATH,
        help="Path to mar_summary.json when --mode summary.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help="Monte Carlo samples per empirical MAR row.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
        help='Torch device, e.g. "cpu", "cuda", or "cuda:0".',
    )
    parser.add_argument(
        "--save-null-samples",
        action="store_true",
        help="Also save every sampled null MAR value to a separate CSV.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def boundary_embed_dim(result_payload: dict[str, Any]) -> int:
    boundaries = result_payload.get("final", {}).get("boundaries") or []
    if not boundaries:
        raise ValueError("Missing final.boundaries in result payload.")
    return int(boundaries[0]["embed_dim"])


def infer_ambient_dim(row: dict[str, Any], results_root: Path) -> int:
    syntax_result = results_root / row["syntax_result"]
    heuristic_result = results_root / row["heuristic_result"]

    syntax_dim = boundary_embed_dim(load_json(syntax_result))
    heuristic_dim = boundary_embed_dim(load_json(heuristic_result))
    if syntax_dim != heuristic_dim:
        raise ValueError(
            "Syntax and heuristic result files disagree on ambient dimension: "
            f"{syntax_result} has {syntax_dim}, {heuristic_result} has {heuristic_dim}"
        )
    return syntax_dim


def mar_from_bases(q_a: torch.Tensor, q_b: torch.Tensor) -> float:
    m_a = q_a.shape[1]
    m_b = q_b.shape[1]
    overlap = q_a.T @ q_b
    mar = (torch.linalg.matrix_norm(overlap, ord="fro") ** 2) / math.sqrt(m_a * m_b)
    return float(mar.item())


def random_orthonormal_basis(
    d: int,
    m: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    gaussian = torch.randn(d, m, generator=generator, device=device, dtype=DTYPE)
    q, _ = torch.linalg.qr(gaussian, mode="reduced")
    return q


def simulate_null_distribution(
    d: int,
    m_syn: int,
    m_heur: int,
    empirical_mar: float,
    *,
    num_samples: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[dict[str, float], list[float]]:
    if not (0 < m_syn <= d and 0 < m_heur <= d):
        raise ValueError(
            f"Invalid dimensions: expected 0 < m_syn,m_heur <= d, got "
            f"d={d}, m_syn={m_syn}, m_heur={m_heur}"
        )

    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            q_syn = random_orthonormal_basis(
                d, m_syn, generator=generator, device=device
            )
            q_heur = random_orthonormal_basis(
                d, m_heur, generator=generator, device=device
            )
            samples.append(mar_from_bases(q_syn, q_heur))

    sample_tensor = torch.tensor(samples, dtype=torch.float64)
    null_mean = float(sample_tensor.mean().item())
    null_std = float(sample_tensor.std(unbiased=True).item())
    z_score = (empirical_mar - null_mean) / null_std if null_std > 0 else math.inf
    p_value = float(norm.sf(z_score))

    return (
        {
            "null_mean": null_mean,
            "null_std": null_std,
            "z_score": float(z_score),
            "right_tailed_p_value": p_value,
        },
        samples,
    )


def rows_from_summary(summary_path: Path) -> list[dict[str, Any]]:
    summary = load_json(summary_path)
    results_root = Path(summary["results_root"])
    rows = []

    for row in summary.get("mar_results", []):
        d = infer_ambient_dim(row, results_root)
        rows.append(
            {
                "label": (
                    f"{row['model']}/{row['direction']}/"
                    f"{row['comparison']}/syn_layer_{row['syntax_layer']}/"
                    f"heur_layer_{row['heuristic_layer']}"
                ),
                "model": row["model"],
                "direction": row["direction"],
                "comparison": row["comparison"],
                "syntax_layer": row["syntax_layer"],
                "heuristic_layer": row["heuristic_layer"],
                "d": d,
                "m_syn": int(row["syntax_m"]),
                "m_heur": int(row["heuristic_m"]),
                "empirical_mar": float(row["mar"]),
                "syntax_result": row["syntax_result"],
                "heuristic_result": row["heuristic_result"],
            }
        )

    return rows


def rows_from_single_config() -> list[dict[str, Any]]:
    return [
        {
            "label": str(SINGLE_CONFIG["label"]),
            "model": "manual",
            "direction": "manual",
            "comparison": "manual",
            "syntax_layer": "",
            "heuristic_layer": "",
            "d": int(SINGLE_CONFIG["d"]),
            "m_syn": int(SINGLE_CONFIG["m_syn"]),
            "m_heur": int(SINGLE_CONFIG["m_heur"]),
            "empirical_mar": float(SINGLE_CONFIG["empirical_mar"]),
            "syntax_result": "",
            "heuristic_result": "",
        }
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    configs = (
        rows_from_summary(args.summary_path)
        if args.mode == "summary"
        else rows_from_single_config()
    )
    if not configs:
        raise SystemExit("No MAR rows found to simulate.")

    stats_rows = []
    sample_rows = []
    for idx, config in enumerate(configs):
        print(
            f"[{idx + 1}/{len(configs)}] {config['label']} "
            f"(d={config['d']}, m_syn={config['m_syn']}, "
            f"m_heur={config['m_heur']}, empirical={config['empirical_mar']:.6g})"
        )
        stats, samples = simulate_null_distribution(
            config["d"],
            config["m_syn"],
            config["m_heur"],
            config["empirical_mar"],
            num_samples=args.num_samples,
            generator=generator,
            device=device,
        )
        stats_rows.append(
            {
                **config,
                "num_samples": args.num_samples,
                "null_mean": stats["null_mean"],
                "null_std": stats["null_std"],
                "z_score": stats["z_score"],
                "right_tailed_p_value": stats["right_tailed_p_value"],
            }
        )

        if args.save_null_samples:
            sample_rows.extend(
                {
                    "label": config["label"],
                    "sample_idx": sample_idx,
                    "null_mar": null_mar,
                }
                for sample_idx, null_mar in enumerate(samples)
            )

    stats_csv = args.output_dir / "mar_random_subspace_null_stats.csv"
    stats_json = args.output_dir / "mar_random_subspace_null_stats.json"
    fieldnames = [
        "label",
        "model",
        "direction",
        "comparison",
        "syntax_layer",
        "heuristic_layer",
        "d",
        "m_syn",
        "m_heur",
        "empirical_mar",
        "num_samples",
        "null_mean",
        "null_std",
        "z_score",
        "right_tailed_p_value",
        "syntax_result",
        "heuristic_result",
    ]
    write_csv(stats_csv, stats_rows, fieldnames)
    with stats_json.open("w", encoding="utf-8") as handle:
        json.dump(stats_rows, handle, indent=2)
    print(f"[saved] {stats_csv}")
    print(f"[saved] {stats_json}")

    if args.save_null_samples:
        samples_csv = args.output_dir / "mar_random_subspace_null_samples.csv"
        write_csv(samples_csv, sample_rows, ["label", "sample_idx", "null_mar"])
        print(f"[saved] {samples_csv}")


if __name__ == "__main__":
    main()
