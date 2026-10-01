#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

try:
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
except ImportError:
    font_manager = None
    plt = None


SCRIPT_DIR = Path(__file__).resolve().parent
BDAS_ROOT = SCRIPT_DIR.parent
CASUAL_INTERVENTION_ROOT = BDAS_ROOT.parent.parent
PYVENE_ROOT = CASUAL_INTERVENTION_ROOT / "pyvene"
if str(PYVENE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYVENE_ROOT))

from pyvene.models.layers import RotateLayer  # noqa: E402


HYPOTHESES = {
    "syn": {
        "label": r"$\mathcal{H}_{syn}$ (Syntax)",
        "sources": [
            ("obj_rel_across_anim", "linear"),
            ("obj_rel_across_anim", "bow"),
        ],
    },
    "prox": {
        "label": r"$\mathcal{H}_{prox}$ (Proximity)",
        "sources": [("obj_rel_across_anim_2", "linear")],
    },
    "maj": {
        "label": r"$\mathcal{H}_{maj}$ (Majority)",
        "sources": [("obj_rel_across_anim_2", "bow")],
    },
}

DIRECTIONS = {
    "source_to_base": {
        "setup": "SG $\\rightarrow$ PL",
        "label": "SG -> PL",
    },
    "base_to_source": {
        "setup": "PL $\\rightarrow$ SG",
        "label": "PL -> SG",
    },
}

MODEL_LABELS = {
    "gemma-3-12b-pt": "Gemma-3-12B",
    "qwen-3-8b": "Qwen-3-8B",
    "olmo-3-7b": "Olmo-3-7B",
    "llama-3.1-8b": "Llama-3.1-8B",
}

MODEL_HIDDEN_DIMS = {
    "gemma-3-12b-pt": 3840,
    "qwen-3-8b": 4096,
    "olmo-3-7b": 4096,
    "llama-3.1-8b": 4096,
}

MODEL_ALIASES = {
    "gemma": "gemma-3-12b-pt",
    "gemma-3-12b": "gemma-3-12b-pt",
    "gemma-3-12b-pt": "gemma-3-12b-pt",
    "qwen": "qwen-3-8b",
    "qwen3": "qwen-3-8b",
    "qwen-3-8b": "qwen-3-8b",
    "olmo": "olmo-3-7b",
    "olmo-3-7b": "olmo-3-7b",
    "llam": "llama-3.1-8b",
    "llama": "llama-3.1-8b",
    "llama-3.1-8b": "llama-3.1-8b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute MAR overlaps between syntax BDAS subspaces and heuristic "
            "BDAS subspaces, and plot learned subspace sizes."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=BDAS_ROOT / "results_das_full_with_saved_boundaries",
        help="Root containing result.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs",
        help="Directory for MAR CSV/JSON files and plots.",
    )
    parser.add_argument(
        "--direction",
        default="all",
        choices=["source_to_base", "base_to_source", "all"],
        help="Direction to analyze. Default: all directions.",
    )
    parser.add_argument(
        "--nua",
        type=int,
        default=4,
        help="Attractor count to analyze.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemma", "qwen", "olmo", "llama"],
        help=(
            "Model aliases or directory names to include, e.g. --models llama olmo. "
            "Use --models all to include every supported model."
        ),
    )
    return parser.parse_args()


def normalize_model_name(model: str) -> str:
    return MODEL_ALIASES.get(model.strip().lower(), model.strip().lower())


def normalize_model_filters(raw_models: list[str]) -> set[str] | None:
    models = []
    for raw_model in raw_models:
        models.extend(part.strip() for part in raw_model.split(","))
    models = [model for model in models if model]
    if not models or any(model.lower() == "all" for model in models):
        return None
    return {normalize_model_name(model) for model in models}


def has_seed_part(result_path: Path, results_root: Path) -> bool:
    return any(part.startswith("seed_") for part in result_path.relative_to(results_root).parts)


def model_family(model: str) -> str:
    normalized = model.lower()
    for family in ("gemma", "qwen", "olmo", "llama"):
        if family in normalized:
            return family
    return normalized.split("-", 1)[0]


def model_sort_key(model: str) -> tuple[int, str]:
    order = {"gemma": 0, "qwen": 1, "olmo": 2, "llama": 3}
    family = model_family(model)
    return (order.get(family, 99), model)


def configure_plot_style() -> None:
    if plt is None or font_manager is None:
        return

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif",
    ]
    font_family = next(
        (font for font in preferred_fonts if font in available_fonts),
        "DejaVu Serif",
    )

    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 16,
            "font.weight": "normal",
            "axes.labelsize": 16,
            "axes.labelweight": "normal",
            "axes.titlesize": 18,
            "axes.titleweight": "normal",
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "figure.titleweight": "normal",
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def boundary_from_result(payload: dict[str, Any]) -> dict[str, float | int]:
    boundaries = payload.get("final", {}).get("boundaries") or []
    if not boundaries:
        raise ValueError("Missing final.boundaries in result payload.")
    boundary = boundaries[0]
    b2 = boundary.get("b2", boundary.get("b2_raw"))
    if b2 is None:
        raise ValueError("Missing b2/b2_raw in final boundary payload.")
    return {
        "embed_dim": int(boundary["embed_dim"]),
        "boundary_raw": float(boundary["boundary_raw"]),
        "boundary_clamped": float(boundary["boundary_clamped"]),
        "b2_raw": float(boundary.get("b2_raw", b2)),
        "b2": float(b2),
        "m_basis": int(math.ceil(float(b2))),
        "m_plot": int(math.floor(float(b2))),
    }


def result_checkpoint_path(payload: dict[str, Any]) -> Path:
    checkpoint = payload.get("training", {}).get("best_rotation_checkpoint", {})
    checkpoint_path = checkpoint.get("path")
    if not checkpoint_path:
        checkpoint_path = payload.get("artifacts", {}).get("best_rotation_checkpoint")
    if not checkpoint_path:
        raise ValueError("Missing best rotation checkpoint path in result payload.")


    print(f"Checkpoint path: {checkpoint_path}")
    return Path(checkpoint_path)


def load_rotation_matrix(checkpoint_path: Path, embed_dim: int) -> torch.Tensor:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    intervention_state = checkpoint.get("intervention_state")
    if not intervention_state:
        raise ValueError(f"Missing intervention_state in {checkpoint_path}")

    rotate_state = intervention_state[0]["rotate_layer_state_dict"]
    if "weight" in rotate_state:
        return rotate_state["weight"].detach().float().cpu()

    rotate_layer = torch.nn.utils.parametrizations.orthogonal(RotateLayer(embed_dim))
    rotate_layer.load_state_dict(rotate_state)
    return rotate_layer.weight.detach().float().cpu()


def basis_from_R_and_b2(R: torch.Tensor, b2: float) -> torch.Tensor:
    """
    R: learned pyvene rotation matrix, shape [embed_dim, embed_dim]
    b2: learned upper boundary in absolute rotated-coordinate units
    """
    R = R.detach().float()
    m = math.ceil(float(b2))
    return R[:, :m]


def mar_from_bases(W_a: torch.Tensor, W_b: torch.Tensor) -> float:
    m_a = W_a.shape[1]
    m_b = W_b.shape[1]
    overlap = W_a.T @ W_b
    mar = (torch.linalg.matrix_norm(overlap, ord="fro") ** 2) / math.sqrt(m_a * m_b)
    return float(mar.item())


def index_results(
    results_root: Path,
    direction: str,
    nua: int,
    models: set[str] | None,
) -> dict[tuple[str, str, str, str, int], dict[str, Any]]:
    indexed = {}
    for result_path in results_root.rglob("result.json"):
        if has_seed_part(result_path, results_root):
            continue

        payload = load_json(result_path)
        model = normalize_model_name(str(payload["model"]["output_dir_name"]))
        if models is not None and model not in models:
            continue

        dataset = str(payload["dataset"]["dataset_name"])
        variation = str(payload["dataset"]["variation"])
        result_direction = str(payload["intervention"]["direction"])
        result_nua = int(payload["dataset"]["nua"])
        layer = int(payload["intervention"]["layer"])

        if direction != "all" and result_direction != direction:
            continue
        if result_nua != nua:
            continue

        key = (model, result_direction, dataset, variation, layer)
        indexed[key] = {
            "result_path": result_path,
            "payload": payload,
            "boundary": boundary_from_result(payload),
            "checkpoint_path": result_checkpoint_path(payload),
        }
    return indexed


def load_basis(entry: dict[str, Any]) -> torch.Tensor:
    boundary = entry["boundary"]
    R = load_rotation_matrix(entry["checkpoint_path"], int(boundary["embed_dim"]))
    return basis_from_R_and_b2(R, float(boundary["b2"]))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_m_by_hypothesis(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    if plt is None:
        print("[warn] matplotlib is not installed; skipping plot.")
        return
    configure_plot_style()

    plot_rows = [row for row in summary_rows if row["hypothesis"] in HYPOTHESES]
    if not plot_rows:
        print("[warn] no m rows available for plotting.")
        return

    panels = [
        (model, direction)
        for model in sorted({row["model"] for row in plot_rows}, key=model_sort_key)
        for direction in DIRECTIONS
        if any(
            row["model"] == model and row["direction"] == direction
            for row in plot_rows
        )
    ]
    if not panels:
        print("[warn] no direction panels available for plotting.")
        return

    hypotheses = ["syn", "prox", "maj"]
    colors = {"syn": "#00A087", "prox": "#E64B35", "maj": "#3C5488"}
    width = 0.20
    group_spacing = 1.05
    x_positions = [idx * group_spacing for idx in range(len(panels))]
    offsets = {"syn": -width, "prox": 0.0, "maj": width}

    fig, ax = plt.subplots(figsize=(max(6.8, 1.35 * len(panels)), 4.6))
    for hypothesis in hypotheses:
        values = []
        full_dimensions = []
        for model, direction in panels:
            matches = [
                row for row in plot_rows
                if (
                    row["model"] == model
                    and row["direction"] == direction
                    and row["hypothesis"] == hypothesis
                )
            ]
            values.append(matches[0]["m"] if matches else float("nan"))
            full_dimensions.append(MODEL_HIDDEN_DIMS[model])

        bar_positions = [x + offsets[hypothesis] for x in x_positions]
        ax.bar(
            bar_positions,
            full_dimensions,
            width=width,
            color=colors[hypothesis],
            alpha=0.15,
            edgecolor="none",
            zorder=1,
        )
        solid_bars = ax.bar(
            bar_positions,
            values,
            width=width,
            label=HYPOTHESES[hypothesis]["label"],
            color=colors[hypothesis],
            edgecolor="black",
            linewidth=1.0,
            zorder=2,
        )
        for bar, value, full_dimension in zip(
            solid_bars, values, full_dimensions
        ):
            if math.isnan(value):
                continue
            percentage = 100.0 * value / full_dimension
            ax.annotate(
                f"{percentage:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                zorder=3,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [
            f"{MODEL_LABELS.get(model, model)}\n({DIRECTIONS[direction]['setup']})"
            for model, direction in panels
        ]
    )
    ax.set_ylabel("Subspace Dimension", labelpad=8)
    ax.set_xlabel("")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_ylim(0, 4500)
    ax.margins(x=0.02)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.97, bottom=0.31)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_filters = normalize_model_filters(args.models)
    indexed = index_results(args.results_root, args.direction, args.nua, model_filters)
    if not indexed:
        raise SystemExit(f"No result.json files found in {args.results_root}")

    m_source_rows = []
    mar_rows = []
    missing = []

    grouped_by_model_direction = defaultdict(dict)
    for key, entry in indexed.items():
        model, direction, dataset, variation, layer = key
        grouped_by_model_direction[(model, direction)][(dataset, variation)] = entry

    for (model, direction), entries in sorted(grouped_by_model_direction.items()):
        syntax_linear = entries.get(("obj_rel_across_anim", "linear"))
        syntax_bow = entries.get(("obj_rel_across_anim", "bow"))
        prox = entries.get(("obj_rel_across_anim_2", "linear"))
        maj = entries.get(("obj_rel_across_anim_2", "bow"))

        for hypothesis, spec in HYPOTHESES.items():
            source_entries = [entries.get(source) for source in spec["sources"]]
            source_entries = [entry for entry in source_entries if entry is not None]
            if not source_entries:
                missing.append(
                    f"{model}/{direction}: missing {hypothesis} sources"
                )
                continue
            mean_m = sum(entry["boundary"]["m_plot"] for entry in source_entries) / len(source_entries)
            m_source_rows.append(
                {
                    "model": model,
                    "direction": direction,
                    "layers": ";".join(
                        str(entry["payload"]["intervention"]["layer"])
                        for entry in source_entries
                    ),
                    "hypothesis": hypothesis,
                    "m": mean_m,
                    "source_count": len(source_entries),
                    "sources": ";".join(
                        str(entry["result_path"].relative_to(args.results_root))
                        for entry in source_entries
                    ),
                }
            )

        comparisons = [
            ("syntax_linear_vs_prox", syntax_linear, prox),
            ("syntax_bow_vs_maj", syntax_bow, maj),
        ]
        for comparison, syntax_entry, heuristic_entry in comparisons:
            if syntax_entry is None or heuristic_entry is None:
                missing.append(f"{model}/{direction}: missing {comparison}")
                continue

            W_syntax = load_basis(syntax_entry)
            W_heuristic = load_basis(heuristic_entry)
            syntax_layer = int(syntax_entry["payload"]["intervention"]["layer"])
            heuristic_layer = int(heuristic_entry["payload"]["intervention"]["layer"])
            mar_rows.append(
                {
                    "model": model,
                    "direction": direction,
                    "syntax_layer": syntax_layer,
                    "heuristic_layer": heuristic_layer,
                    "comparison": comparison,
                    "mar": mar_from_bases(W_syntax, W_heuristic),
                    "syntax_m": W_syntax.shape[1],
                    "heuristic_m": W_heuristic.shape[1],
                    "syntax_result": str(
                        syntax_entry["result_path"].relative_to(args.results_root)
                    ),
                    "heuristic_result": str(
                        heuristic_entry["result_path"].relative_to(args.results_root)
                    ),
                    "syntax_checkpoint": str(syntax_entry["checkpoint_path"]),
                    "heuristic_checkpoint": str(heuristic_entry["checkpoint_path"]),
                }
            )

    mar_csv = args.output_dir / "mar_results.csv"
    write_csv(
        mar_csv,
        mar_rows,
        [
            "model",
            "direction",
            "syntax_layer",
            "heuristic_layer",
            "comparison",
            "mar",
            "syntax_m",
            "heuristic_m",
            "syntax_result",
            "heuristic_result",
            "syntax_checkpoint",
            "heuristic_checkpoint",
        ],
    )
    print(f"[saved] {mar_csv}")

    m_csv = args.output_dir / "boundary_m_by_hypothesis.csv"
    write_csv(
        m_csv,
        m_source_rows,
        ["model", "direction", "layers", "hypothesis", "m", "source_count", "sources"],
    )
    print(f"[saved] {m_csv}")

    summary_path = args.output_dir / "mar_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "results_root": str(args.results_root),
                "direction": args.direction,
                "nua": args.nua,
                "mar_results": mar_rows,
                "boundary_m_by_hypothesis": m_source_rows,
                "missing": missing,
            },
            handle,
            indent=2,
        )
    print(f"[saved] {summary_path}")

    plot_path = args.output_dir / "boundary_m_by_model_hypothesis.pdf"
    plot_m_by_hypothesis(m_source_rows, plot_path)

    if missing:
        print("[warn] missing inputs:")
        for item in missing:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
