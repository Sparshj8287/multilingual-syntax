import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Min-K%++ familiarity and logit-difference syntax confidence "
            "from results_<N>.jsonl files and generate per-file scatter plots."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b-pt",
        help="Model checkpoint name or local path used in evaluation.",
    )
    parser.add_argument(
        "--use-prompt",
        action="store_true",
        help="Read from prompt-mode result directory (<model_id>-prompt).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=script_dir / "results",
        help="Root directory containing results/<model_id>/results_<N>.jsonl.",
    )
    parser.add_argument(
        "--plots-root",
        type=Path,
        default=script_dir / "plots",
        help="Root directory for plots. Output is plots/<model_id>/mink_vs_logit_diff_<N>.png.",
    )
    parser.add_argument(
        "--min-attractors",
        type=int,
        default=1,
        help="Minimum number of attractors (inclusive).",
    )
    parser.add_argument(
        "--max-attractors",
        type=int,
        default=6,
        help="Maximum number of attractors (inclusive).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Accuracy vs. Familiarity in Subject-Verb Agreement (NUA={n})",
        help="Plot title template. Use {n} for attractor count.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def flatten_results(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    flat_rows: List[Dict[str, Any]] = []
    for row in rows:
        nua = row.get("NUA")
        flat_rows.append(
            {
                "variant": "base",
                "variant_label": "base_sentence",
                "NUA": nua,
                "mink_score": row.get("base_mink_score"),
                "logit_diff": row.get("base_logit_diff"),
            }
        )
        flat_rows.append(
            {
                "variant": "source",
                "variant_label": "source_sentence",
                "NUA": nua,
                "mink_score": row.get("source_mink_score"),
                "logit_diff": row.get("source_logit_diff"),
            }
        )

    df = pd.DataFrame(flat_rows)
    df["mink_score"] = pd.to_numeric(df["mink_score"], errors="coerce")
    df["logit_diff"] = pd.to_numeric(df["logit_diff"], errors="coerce")
    df["NUA"] = pd.to_numeric(df["NUA"], errors="coerce")
    df = df.dropna(subset=["mink_score", "logit_diff", "NUA"]).copy()
    df["NUA"] = df["NUA"].astype(int)
    return df


def infer_model_id(model_name: str, use_prompt: bool = False) -> str:
    model_id = Path(model_name.rstrip("/")).name
    model_id = model_id.replace(" ", "_")
    if use_prompt:
        model_id = f"{model_id}-prompt"
    return model_id


def safe_corr(series_x: pd.Series, series_y: pd.Series) -> tuple[float, float]:
    try:
        return pearsonr(series_x, series_y)
    except ValueError:
        return float("nan"), float("nan")


def safe_rank_corr(series_x: pd.Series, series_y: pd.Series) -> tuple[float, float]:
    try:
        return spearmanr(series_x, series_y)
    except ValueError:
        return float("nan"), float("nan")


def fmt_corr(metric: str, value: float, p_val: float) -> str:
    if pd.isna(value) or pd.isna(p_val):
        return f"{metric}: N/A"
    return f"{metric}: {value:.3f} (p={p_val:.2e})"


def main() -> None:
    args = parse_args()
    if args.min_attractors > args.max_attractors:
        raise ValueError("--min-attractors must be <= --max-attractors")

    model_id = infer_model_id(args.model, use_prompt=args.use_prompt)
    model_results_dir = args.results_root / model_id
    model_plots_dir = args.plots_root / model_id
    model_plots_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    variant_palette = {
        "base_sentence": "#1f77b4",
        "source_sentence": "#d62728",
    }

    for attractor_n in range(args.min_attractors, args.max_attractors + 1):
        input_jsonl = model_results_dir / f"results_{attractor_n}.jsonl"
        output_plot = model_plots_dir / f"mink_vs_logit_diff_{attractor_n}.png"

        if not input_jsonl.exists():
            print(f"Skipping missing result file: {input_jsonl}")
            continue

        rows = load_jsonl(input_jsonl)
        df = flatten_results(rows)
        if len(df) < 2:
            print(f"Skipping attractor={attractor_n}: not enough valid rows.")
            continue

        pearson_r, pearson_p = safe_corr(df["mink_score"], df["logit_diff"])
        spearman_rho, spearman_p = safe_rank_corr(df["mink_score"], df["logit_diff"])

        base_df = df[df["variant"] == "base"]
        source_df = df[df["variant"] == "source"]
        base_pearson, base_pearson_p = safe_corr(
            base_df["mink_score"], base_df["logit_diff"]
        )
        base_spearman, base_spearman_p = safe_rank_corr(
            base_df["mink_score"], base_df["logit_diff"]
        )
        source_pearson, source_pearson_p = safe_corr(
            source_df["mink_score"], source_df["logit_diff"]
        )
        source_spearman, source_spearman_p = safe_rank_corr(
            source_df["mink_score"], source_df["logit_diff"]
        )

        print(f"Correlation Results (NUA={attractor_n})")
        print("--------------------------------")
        print(f"Pearson r:       {pearson_r:.6f}")
        print(f"Pearson p-value: {pearson_p:.6e}")
        print(f"Spearman rho:    {spearman_rho:.6f}")
        print(f"Spearman p-value:{spearman_p:.6e}")

        fig, ax = plt.subplots(figsize=(11, 8))
        sns.scatterplot(
            data=df,
            x="mink_score",
            y="logit_diff",
            hue="variant_label",
            style="variant_label",
            palette=variant_palette,
            alpha=0.85,
            s=60,
            ax=ax,
        )
        ax.axhline(0.0, linestyle="--", linewidth=1.3, color="black")
        ax.set_xlabel("Min-K%++ Score (Familiarity)")
        ax.set_ylabel("Logit Difference (Syntax Confidence)")
        ax.set_title(args.title.replace("{n}", str(attractor_n)))
        ax.legend(title="Sentence Variant", loc="best")

        stats_text = (
            f"{fmt_corr('Overall Pearson r', pearson_r, pearson_p)}\n"
            f"{fmt_corr('Overall Spearman rho', spearman_rho, spearman_p)}\n"
            f"{fmt_corr('Base Pearson r', base_pearson, base_pearson_p)}\n"
            f"{fmt_corr('Base Spearman rho', base_spearman, base_spearman_p)}\n"
            f"{fmt_corr('Source Pearson r', source_pearson, source_pearson_p)}\n"
            f"{fmt_corr('Source Spearman rho', source_spearman, source_spearman_p)}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
        )

        fig.tight_layout()
        fig.savefig(output_plot, dpi=300)
        plt.close(fig)
        print(f"Saved plot to: {output_plot}")


if __name__ == "__main__":
    main()
