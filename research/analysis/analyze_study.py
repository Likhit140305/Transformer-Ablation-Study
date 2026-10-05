"""Create a publication-ready, seed-level summary from schema-v2 run outputs.

This script intentionally rejects ambiguous inputs.  A result table with
duplicated seeds or incompatible training budgets is worse than no table.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def load_runs(results_root: Path) -> list[dict]:
    runs: list[dict] = []
    for path in results_root.glob("**/results.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("schema_version") != 2:
            continue
        required = {"condition_signature", "resolved_model", "seed", "best_val_ppl"}
        missing = required - result.keys()
        if missing:
            raise ValueError(f"{path} is missing required provenance fields: {sorted(missing)}")
        result["result_path"] = str(path)
        runs.append(result)
    if not runs:
        raise ValueError("No schema-v2 runs found. Historical schema-v1 outputs are intentionally excluded.")
    return runs


def grouped_runs(runs: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        groups[run["condition_signature"]].append(run)
    for signature, group in groups.items():
        seeds = [run["seed"] for run in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Condition {signature} contains duplicate seeds: {seeds}")
        reference = group[0]
        for run in group[1:]:
            for key in ("experiment", "resolved_model", "dataset_name"):
                if run.get(key) != reference.get(key):
                    raise ValueError(f"Condition signature collision for {signature} on field {key}")
    return dict(groups)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 10_000) -> tuple[float, float]:
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def summarize(groups: dict[str, list[dict]], baseline: str) -> list[dict]:
    if baseline not in groups:
        raise ValueError(f"Baseline signature {baseline!r} was not found.")
    rng = np.random.default_rng(20260802)
    baseline_values = np.asarray([r["best_val_ppl"] for r in groups[baseline]], dtype=float)
    rows = []
    for signature, group in sorted(groups.items()):
        values = np.asarray([r["best_val_ppl"] for r in group], dtype=float)
        ci_low, ci_high = bootstrap_ci(values, rng)
        row = {
            "condition_signature": signature,
            "experiment": group[0]["experiment"],
            "n_seeds": len(values),
            "mean_val_ppl": float(values.mean()),
            "std_val_ppl": float(values.std(ddof=1)) if len(values) > 1 else None,
            "mean_95ci": [ci_low, ci_high],
            "seeds": sorted(int(r["seed"]) for r in group),
            "resolved_model": group[0]["resolved_model"],
            "delta_vs_baseline": float(values.mean() - baseline_values.mean()),
        }
        if signature != baseline:
            test = stats.ttest_ind(values, baseline_values, equal_var=False)
            row["welch_t"] = float(test.statistic)
            row["welch_p"] = float(test.pvalue)
        else:
            row["welch_t"] = None
            row["welch_p"] = None
        rows.append(row)
    return rows


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    table = [
        "| Experiment | Condition | n | Validation PPL (mean, 95% bootstrap CI) | Δ vs baseline | Welch p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        low, high = row["mean_95ci"]
        p_value = "—" if row["welch_p"] is None else f"{row['welch_p']:.3g}"
        table.append(
            f"| {row['experiment']} | `{row['condition_signature']}` | {row['n_seeds']} | "
            f"{row['mean_val_ppl']:.3f} [{low:.3f}, {high:.3f}] | "
            f"{row['delta_vs_baseline']:+.3f} | {p_value} |"
        )
    caveat = (
        "\n\nNegative ΔPPL means lower (better) validation perplexity. Welch p-values are exploratory "
        "unless the comparison set and correction procedure were specified before inspecting results. "
        "Never use this validation table as a test-set result.\n"
    )
    (output_dir / "results_table.md").write_text("\n".join(table) + caveat, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("research/results"))
    parser.add_argument("--baseline-signature", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("research/analysis/output"))
    args = parser.parse_args()
    rows = summarize(grouped_runs(load_runs(args.results_root)), args.baseline_signature)
    write_outputs(rows, args.output_dir)
    print(f"Wrote {len(rows)} conditions to {args.output_dir}")


if __name__ == "__main__":
    main()
