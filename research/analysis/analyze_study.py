"""Turn schema-v3 run outputs into the paper's tables, figures and statistics.

Primary metric: test cross-entropy (nats/token) of the best-validation checkpoint;
perplexity = exp(loss). For each ablation and context length we test the
difference from the baseline at the same length (Welch t-test, Holm-corrected
over all 14 comparisons). The interaction contrast asks whether an ablation's
effect changes with context: (abl_512 - base_512) - (abl_128 - base_128), on the
loss scale so that it is a ratio of perplexity ratios (Holm over 7 contrasts).

    python research/analysis/analyze_study.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.datasets.wikitext import load_wikitext_datasets

SCHEMA_VERSION = 3
LABELS = {
    "baseline": "Baseline",
    "abl_no_pos_enc": "No positional enc.",
    "abl_sinusoidal_pos": "Sinusoidal pos. enc.",
    "abl_single_head": "Single head",
    "abl_reduced_ffn": r"FFN width $1\times$",
    "abl_linear_ffn": "Linear FFN",
    "abl_no_residual": "No residual",
    "abl_post_layernorm": "Post-LN",
}
ORDER = list(LABELS)
BUDGET_CONTROL = "baseline_2x_budget"
# Reference categorical palette, one fixed colour per condition (ORDER); slots 7-8 swapped so the
# Post-LN curve (violet) stays distinct from No-pos-enc (orange) when No-residual is not drawn.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#e34948", "#4a3aa7"]
INK, MUTED = "#0b0b0b", "#52514e"


def load_runs(results_root: Path) -> dict[tuple[str, int], list[dict]]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for path in results_root.glob("*/L*/*/results.json"):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("schema_version") == SCHEMA_VERSION:
            groups[(run["experiment"], run["seq_length"])].append(run)
    for key, runs in groups.items():
        seeds = [r["seed"] for r in runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{key} has duplicate seeds {seeds}; archive the repeat explicitly")
        if len({r["condition_signature"] for r in runs}) != 1:
            raise ValueError(f"{key} mixes runs with different configurations")
    if not groups:
        raise ValueError(f"No schema-v{SCHEMA_VERSION} runs under {results_root}")
    return groups


def balanced(groups: dict[tuple[str, int], list[dict]]) -> tuple[dict, list[int]]:
    """Keep only seeds finished by every condition, so all cells share the same seeds."""
    seeds = sorted(set.intersection(*({r["seed"] for r in runs} for runs in groups.values())))
    if len(seeds) < 2:
        raise ValueError("Need at least two seeds completed by every condition")
    return {key: sorted((r for r in runs if r["seed"] in seeds), key=lambda r: r["seed"]) for key, runs in groups.items()}, seeds


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted, running = np.empty(len(p_values)), 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def welch_difference(a: np.ndarray, b: np.ndarray) -> dict:
    """Mean(a) - mean(b) with a Welch 95% CI and two-sided p-value."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1)) if se > 0 else float("inf")
    diff = float(a.mean() - b.mean())
    half = stats.t.ppf(0.975, df) * se
    p = float(stats.ttest_ind(a, b, equal_var=False).pvalue) if se > 0 else 0.0
    return {"diff": diff, "ci": [diff - half, diff + half], "p": p}


def contrast(groups: dict, ablation: str, lengths: tuple[int, int]) -> dict:
    """(abl_L2 - base_L2) - (abl_L1 - base_L1) with Welch-Satterthwaite df over four independent cells."""
    signs = {(ablation, lengths[1]): 1, ("baseline", lengths[1]): -1, (ablation, lengths[0]): -1, ("baseline", lengths[0]): 1}
    cells = {k: np.array([r["test_loss"] for r in groups[k]]) for k in signs}
    estimate = sum(s * cells[k].mean() for k, s in signs.items())
    variances = [cells[k].var(ddof=1) / len(cells[k]) for k in signs]
    se = math.sqrt(sum(variances))
    df = sum(variances) ** 2 / sum(v**2 / (len(cells[k]) - 1) for v, k in zip(variances, signs))
    t = estimate / se
    half = stats.t.ppf(0.975, df) * se
    return {"diff": float(estimate), "ci": [estimate - half, estimate + half], "p": float(2 * stats.t.sf(abs(t), df))}


def summarize(groups: dict, lengths: list[int]) -> dict:
    cells, comparisons = {}, []
    for (experiment, length), runs in groups.items():
        loss = np.array([r["test_loss"] for r in runs])
        cells[f"{experiment}@{length}"] = {
            "experiment": experiment,
            "seq_length": length,
            "n": len(runs),
            "params": runs[0]["total_params"],
            "test_ppl_mean": float(np.exp(loss).mean()),
            "test_ppl_sd": float(np.exp(loss).std(ddof=1)),
            "val_ppl_mean": float(np.mean([r["best_val_ppl"] for r in runs])),
            "test_loss_mean": float(loss.mean()),
            "best_epoch_mean": float(np.mean([r["best_epoch"] for r in runs])),
            "diverged": int(sum(r["diverged"] for r in runs)),
            "runtime_min": float(np.mean([r["runtime_seconds"] for r in runs]) / 60),
        }
    for length in lengths:
        base = np.array([r["test_loss"] for r in groups[("baseline", length)]])
        for experiment in ORDER[1:]:
            if (experiment, length) in groups:
                result = welch_difference(np.array([r["test_loss"] for r in groups[(experiment, length)]]), base)
                comparisons.append({"experiment": experiment, "seq_length": length, **result})
    for row, p in zip(comparisons, holm([c["p"] for c in comparisons])):
        row["p_holm"] = p
        row["ppl_ratio_pct"] = 100 * (math.exp(row["diff"]) - 1)
    interactions = []
    if len(lengths) == 2:
        interactions = [
            {"experiment": e, **contrast(groups, e, tuple(lengths))}
            for e in ORDER[1:]
            if all((x, L) in groups for x in (e, "baseline") for L in lengths)
        ]
        for row, p in zip(interactions, holm([r["p"] for r in interactions])):
            row["p_holm"] = p
    return {"cells": cells, "comparisons": comparisons, "interactions": interactions}


def unigram_test_ppl(run: dict) -> float:
    """Test PPL of a train-split unigram model: the floor a context-free predictor reaches."""
    d = run["config"]["dataset"]
    train, _, test, vocabulary = load_wikitext_datasets(
        data_dir=ROOT / "research" / "data", dataset_dir=d["dataset_dir"],
        files={"train": d["train_file"], "validation": d["validation_file"], "test": d["test_file"]},
        seq_length=d["seq_length"], pad_token=d["pad_token"], unk_token=d["unk_token"],
        max_vocab_size=d["max_vocab_size"], min_frequency=d["min_frequency"], verify_checksums=False,
    )
    log_p = torch.bincount(train.tokens, minlength=vocabulary.size).double().add(1e-9).log()
    log_p -= torch.logsumexp(log_p, 0)
    return math.exp(-log_p[test.tokens].mean().item())


def budget_control(all_groups: dict, groups: dict, lengths: list[int]) -> dict:
    """Baseline at 2x epochs vs 1x on the same seeds: does the long-context gap close with more training?"""
    out = {}
    for L in lengths:
        control = all_groups.get((BUDGET_CONTROL, L), [])
        if len(control) < 2:
            continue
        seeds = {r["seed"] for r in control}
        base = [r for r in groups[("baseline", L)] if r["seed"] in seeds]
        ppl = np.exp([r["test_loss"] for r in control])
        out[L] = {
            "n": len(control),
            "test_ppl_mean": float(ppl.mean()),
            "test_ppl_sd": float(ppl.std(ddof=1)),
            "best_epoch_mean": float(np.mean([r["best_epoch"] for r in control])),
            "baseline_same_seeds_ppl": float(np.exp([r["test_loss"] for r in base]).mean()),
        }
    return out


def fmt_p(p: float) -> str:
    return r"$<$0.001" if p < 1e-3 else f"{p:.3f}"


def write_tables(summary: dict, lengths: list[int], seeds: list[int], out: Path, unigram: float) -> None:
    cells, comp = summary["cells"], {(c["experiment"], c["seq_length"]): c for c in summary["comparisons"]}
    lines = [
        r"\begin{tabular}{l r " + " ".join(["r r r"] * len(lengths)) + "}",
        r"\toprule",
        " & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{$L={L}$}}" for L in lengths) + r" \\",
        " ".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(lengths))),
        "Condition & Params & " + " & ".join([r"Test PPL & $\Delta$\% & $p_\mathrm{Holm}$"] * len(lengths)) + r" \\",
        r"\midrule",
    ]
    for experiment in ORDER:
        row = [LABELS[experiment], f"{cells[f'{experiment}@{lengths[0]}']['params'] / 1e6:.2f}M"]
        for L in lengths:
            cell = cells[f"{experiment}@{L}"]
            row.append(f"{cell['test_ppl_mean']:.1f} $\\pm$ {cell['test_ppl_sd']:.1f}")
            if experiment == "baseline":
                row += ["--", "--"]
            else:
                c = comp[(experiment, L)]
                row += [f"{c['ppl_ratio_pct']:+.1f}", fmt_p(c["p_holm"])]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "main_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if summary["interactions"]:
        lines = [
            r"\begin{tabular}{l r r r}", r"\toprule",
            r"Ablation & Interaction (nats) & 95\% CI & $p_\mathrm{Holm}$ \\", r"\midrule",
        ]
        for row in summary["interactions"]:
            lines.append(
                f"{LABELS[row['experiment']]} & {row['diff']:+.3f} & "
                f"[{row['ci'][0]:+.3f}, {row['ci'][1]:+.3f}] & {fmt_p(row['p_holm'])} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        (out / "interaction.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    macros = {
        "NSeeds": str(len(seeds)),
        "BaselinePPLShort": f"{cells[f'baseline@{lengths[0]}']['test_ppl_mean']:.1f}",
        "BaselinePPLLong": f"{cells[f'baseline@{lengths[-1]}']['test_ppl_mean']:.1f}",
        "BaselineSDShort": f"{cells[f'baseline@{lengths[0]}']['test_ppl_sd']:.2f}",
        "UnigramTestPPL": f"{unigram:.0f}",
    }
    for L, row in summary.get("budget_control", {}).items():
        suffix = "Short" if L == lengths[0] else "Long"
        macros[f"BudgetPPL{suffix}"] = f"{row['test_ppl_mean']:.1f}"
        macros[f"BudgetSD{suffix}"] = f"{row['test_ppl_sd']:.1f}"
        macros[f"BudgetEpoch{suffix}"] = f"{row['best_epoch_mean']:.1f}"
        macros[f"BudgetBasePPL{suffix}"] = f"{row['baseline_same_seeds_ppl']:.1f}"
    (out / "macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()), encoding="utf-8")


def style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK, width=0.6, labelsize=7)


def log_ticks(axis) -> None:
    """Label a log axis with ratio ticks (0.9x, 1x, 1.25x, ...) instead of powers of ten."""
    candidates = [1, 2, 1.5, 4, 1.25, 3, 8, 1.1, 0.9, 6, 16, 0.8, 1.05, 0.95, 12]  # in order of preference
    axis.axes.autoscale_view()
    lo, hi = axis.get_view_interval()
    min_gap = 0.13 * math.log(hi / lo)  # keep labels at least 13% of the axis apart
    ticks: list[float] = []
    for t in candidates:
        if lo <= t <= hi and all(abs(math.log(t / u)) >= min_gap for u in ticks):
            ticks.append(t)
    axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" + r"$\times$"))
    axis.set_minor_locator(matplotlib.ticker.NullLocator())


def plot_effects(summary: dict, lengths: list[int], out: Path) -> None:
    """Dot plot of the perplexity change vs baseline, with 95% CIs, per context length."""
    experiments = ORDER[1:]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    offsets = np.linspace(-0.17, 0.17, len(lengths))
    for color, offset, L in zip(PALETTE, offsets, lengths):
        rows = {c["experiment"]: c for c in summary["comparisons"] if c["seq_length"] == L}
        y = np.arange(len(experiments)) + offset
        ratio = np.array([math.exp(rows[e]["diff"]) for e in experiments])
        lo = np.array([math.exp(rows[e]["ci"][0]) for e in experiments])
        hi = np.array([math.exp(rows[e]["ci"][1]) for e in experiments])
        ax.errorbar(ratio, y, xerr=[ratio - lo, hi - ratio], fmt="o", ms=3.5, color=color, ecolor=color,
                    elinewidth=1.2, capsize=0, label=f"$L={L}$", mec="white", mew=0.6, zorder=3)
    ax.axvline(1.0, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.set_xscale("log")
    ax.set_yticks(range(len(experiments)), [LABELS[e] for e in experiments])
    ax.invert_yaxis()
    ax.set_xlabel("Test PPL relative to baseline (log scale)", fontsize=7, color=INK)
    log_ticks(ax.xaxis)
    ax.grid(axis="x", color="#e4e3df", lw=0.5, zorder=0)
    style_axes(ax)
    ax.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(lengths), handletextpad=0.2)
    fig.tight_layout()
    fig.savefig(out / "effects.pdf")
    plt.close(fig)


def plot_curves(groups: dict, lengths: list[int], out: Path) -> None:
    """Per-epoch validation PPL of each ablation relative to the baseline (seed means), one panel per length."""
    fig, axes = plt.subplots(1, len(lengths), figsize=(7.0, 2.4), sharey=True)
    for ax, L in zip(np.atleast_1d(axes), lengths):
        base = np.mean([r["history"]["val_loss"] for r in groups[("baseline", L)]], axis=0)
        ax.axhline(1.0, color=PALETTE[0], lw=1.2, label=LABELS["baseline"])
        for color, experiment in zip(PALETTE[1:], ORDER[1:]):
            if experiment == "abl_no_residual":  # 3-8x off scale throughout; shown in the effects plot instead
                continue
            gap = np.mean([r["history"]["val_loss"] for r in groups[(experiment, L)]], axis=0) - base
            ax.plot(np.arange(1, len(gap) + 1), np.exp(gap), color=color, lw=1.2, label=LABELS[experiment])
        ax.set_yscale("log")
        ax.set_title(f"$L={L}$", fontsize=8, color=INK)
        ax.set_xlabel("Epoch", fontsize=7, color=INK)
        ax.set_xlim(1, len(base))
        ax.grid(axis="y", color="#e4e3df", lw=0.5, which="both")
        style_axes(ax)
    axes[0].set_ylabel("Val. PPL / baseline (log)", fontsize=7, color=INK)
    log_ticks(axes[0].yaxis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.5, frameon=False, loc="center left", bbox_to_anchor=(0.80, 0.5))
    fig.tight_layout(rect=(0, 0, 0.80, 1))
    fig.savefig(out / "curves.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=ROOT / "research" / "results")
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "research" / "paper")
    args = parser.parse_args()

    all_groups = load_runs(args.results_root)
    groups, seeds = balanced({k: v for k, v in all_groups.items() if k[0] in LABELS})
    lengths = sorted({L for _, L in groups})
    missing = [(e, L) for e in ORDER for L in lengths if (e, L) not in groups]
    if missing:
        raise ValueError(f"Incomplete grid, missing cells: {missing}")
    summary = summarize(groups, lengths)
    summary["seeds"] = seeds

    tables, figures = args.paper_dir / "tables", args.paper_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    (args.paper_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["unigram_test_ppl"] = unigram_test_ppl(groups[("baseline", lengths[0])][0])
    summary["budget_control"] = budget_control(all_groups, groups, lengths)
    write_tables(summary, lengths, seeds, tables, summary["unigram_test_ppl"])
    plot_effects(summary, lengths, figures)
    plot_curves(groups, lengths, figures)

    print(f"seeds={seeds}  lengths={lengths}")
    for c in summary["comparisons"]:
        print(f"{c['experiment']:>20} L={c['seq_length']:<4} dPPL={c['ppl_ratio_pct']:+7.1f}%  p_holm={c['p_holm']:.3g}")
    for L, row in summary["budget_control"].items():
        print(f"budget control L={L}: 2x {row['test_ppl_mean']:.2f}+/-{row['test_ppl_sd']:.2f} (best ep {row['best_epoch_mean']:.1f}) vs 1x {row['baseline_same_seeds_ppl']:.2f}")
    for c in summary["interactions"]:
        print(f"{c['experiment']:>20} interaction={c['diff']:+.4f} nats CI=[{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}] p_holm={c['p_holm']:.3g}")


if __name__ == "__main__":
    main()
