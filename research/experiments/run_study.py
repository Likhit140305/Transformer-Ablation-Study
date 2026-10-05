import json
import logging
import math
import subprocess
import os
import sys
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("StudyRunner")

def check_cuda_available():
    try:
        import torch
    except ImportError:
        raise RuntimeError("torch is not importable in this environment; cannot verify CUDA availability.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested (--device cuda) but torch.cuda.is_available() is False. "
            "Check drivers/CUDA_VISIBLE_DEVICES/your torch build before running experiments."
        )
    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)} (device count={torch.cuda.device_count()})")

def run_experiment(experiment_config: str, seed: int, device: str, additional_args: list = None):
    args = [
        sys.executable, "research/experiments/train.py",
        f"experiment={experiment_config}",
        f"seed={seed}",
        f"runtime.device={device}",
    ]
    if additional_args:
        args.extend(additional_args)

    logger.info(f"Running: {' '.join(args)}")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Experiment failed: {result.stderr}")
        raise RuntimeError(f"Experiment {experiment_config} seed {seed} failed.")
    return result

def collect_results(experiment_name: str, seeds: list, *, seq_length: int, max_epochs: int):
    """Collect one schema-v2 result per seed for an exactly specified condition.

    Historical files are deliberately excluded: earlier versions did not record
    resolved ablations and mixed runs with different training budgets.
    """
    results_dir = Path("research/results") / experiment_name
    by_seed = {}

    if not results_dir.exists():
        return None

    for d in results_dir.iterdir():
        if d.is_dir():
            res_file = d / "results.json"
            if res_file.exists():
                with open(res_file, "r") as f:
                    data = json.load(f)
                cfg = data.get("config", {})
                is_requested_condition = (
                    data.get("schema_version") == 2
                    and data.get("data_source") == "local_verified_wikitext"
                    and data.get("experiment") == experiment_name
                    and data.get("seed") in seeds
                    and cfg.get("dataset", {}).get("seq_length") == seq_length
                    and cfg.get("training", {}).get("max_epochs") == max_epochs
                )
                if is_requested_condition:
                    seed = data["seed"]
                    if seed in by_seed:
                        raise RuntimeError(
                            f"Duplicate completed runs for {experiment_name}, seed={seed}, "
                            f"seq_length={seq_length}, max_epochs={max_epochs}. "
                            "Do not treat repeat runs as independent seeds; archive one explicitly."
                        )
                    by_seed[seed] = data["best_val_ppl"]

    missing = sorted(set(seeds) - set(by_seed))
    if missing:
        raise RuntimeError(
            f"Missing schema-v2 results for {experiment_name}; required seeds: {missing}. "
            "Rerun the exact condition rather than mixing historical output."
        )

    ppls = [by_seed[seed] for seed in seeds]
    if not ppls:
        return None

    mean_ppl = sum(ppls) / len(ppls)
    variance = sum((x - mean_ppl) ** 2 for x in ppls) / (len(ppls) - 1) if len(ppls) > 1 else 0
    std_dev = math.sqrt(variance)

    return {"mean": mean_ppl, "std": std_dev, "n": len(ppls), "raw": ppls}

def find_completed_run(experiment_name: str, seed: int, seq_length: int, max_epochs: int):
    """Look for an existing results.json matching this exact (experiment, seed, seq_length, max_epochs).
    Only used for phase 4, where seq_length/max_epochs are explicitly varied and known."""
    results_dir = Path("research/results") / experiment_name
    if not results_dir.exists():
        return None
    for d in results_dir.iterdir():
        if not d.is_dir():
            continue
        res_file = d / "results.json"
        if not res_file.exists():
            continue
        with open(res_file, "r") as f:
            data = json.load(f)
        cfg = data.get("config", {})
        if (data.get("seed") == seed
                and cfg.get("dataset", {}).get("seq_length") == seq_length
                and cfg.get("training", {}).get("max_epochs") == max_epochs
                and data.get("schema_version") == 2):
            return data
    return None

def phase1_variance_pilot(device):
    logger.info("=== PHASE 1: Variance Pilot ===")
    seeds = [42, 1337, 2023, 7, 31415]
    for seed in seeds:
        run_experiment("baseline", seed, device)

    stats = collect_results("baseline", seeds, seq_length=128, max_epochs=30)
    logger.info(f"Baseline Stats: Mean PPL = {stats['mean']:.4f}, Std = {stats['std']:.4f}")
    return stats

def phase3_screening(baseline_stats, device):
    logger.info("=== PHASE 3: Architecture Screening ===")
    ablations = [
        "abl_no_pos_enc",
        "abl_sinusoidal_pos",
        "abl_single_head",
        "abl_reduced_ffn",
        "abl_no_residual",
        "abl_post_layernorm",
        "abl_linear_ffn"
    ]
    # Screening is exploratory but uses an internally matched baseline.
    # Promotion is not a significance claim; confirmation needs fresh seeds.
    seeds = [42, 1337, 2023]
    promoted = []

    for seed in seeds:
        run_experiment("baseline", seed, device, ["training.max_epochs=5"])
    screening_baseline = collect_results("baseline", seeds, seq_length=128, max_epochs=5)
    threshold = 3 * screening_baseline["std"]

    for abl in ablations:
        for seed in seeds:
            run_experiment(abl, seed, device, ["training.max_epochs=5"])

        stats = collect_results(abl, seeds, seq_length=128, max_epochs=5)
        if stats:
            delta = abs(stats["mean"] - screening_baseline["mean"])
            logger.info(f"{abl}: Mean PPL = {stats['mean']:.4f} | Delta = {delta:.4f} (Threshold: {threshold:.4f})")

            if delta > threshold:
                logger.info(f"--> PROMOTED: {abl}")
                promoted.append(abl)
            else:
                logger.info(f"--> REJECTED: {abl}")

    return promoted

def recompute_promoted_from_disk(device):
    """Re-derive baseline_stats and the promoted list from results already on disk,
    without rerunning phase 1 or phase 3. Mirrors phase3_screening's own threshold logic."""
    baseline_stats = collect_results("baseline", [42, 1337, 2023, 7, 31415], seq_length=128, max_epochs=30)
    logger.info(f"Recovered Baseline Stats from disk: Mean PPL = {baseline_stats['mean']:.4f}, Std = {baseline_stats['std']:.4f}")

    ablations = [
        "abl_no_pos_enc",
        "abl_sinusoidal_pos",
        "abl_single_head",
        "abl_reduced_ffn",
        "abl_no_residual",
        "abl_post_layernorm",
        "abl_linear_ffn"
    ]
    screening_baseline = collect_results("baseline", [42, 1337, 2023], seq_length=128, max_epochs=5)
    threshold = 3 * screening_baseline["std"]
    promoted = []
    for abl in ablations:
        stats = collect_results(abl, [42, 1337, 2023], seq_length=128, max_epochs=5)
        if stats is None:
            logger.warning(f"No phase 3 results found on disk for {abl} — treating as not promoted. "
                            f"If phase 3 actually ran this ablation, check research/results/{abl}/ exists.")
            continue
        delta = abs(stats["mean"] - screening_baseline["mean"])
        if delta > threshold:
            logger.info(f"--> PROMOTED (from disk): {abl} | delta={delta:.4f} > threshold={threshold:.4f}")
            promoted.append(abl)
        else:
            logger.info(f"--> REJECTED (from disk): {abl} | delta={delta:.4f} <= threshold={threshold:.4f}")

    return baseline_stats, promoted

def phase4_interaction(promoted_ablations, device):
    logger.info("=== PHASE 4: Sequence-Length Interaction ===")
    # This is the confirmatory stage: use new, adequately replicated full-budget
    # runs.  Do not report the short screening runs as final evidence.
    seeds = [42, 1337, 2023, 7, 31415]
    lengths = [128, 512]
    max_epochs = 30

    configs_to_run = ["baseline"] + promoted_ablations

    for config in configs_to_run:
        for length in lengths:
            for seed in seeds:
                existing = find_completed_run(config, seed, length, max_epochs)
                if existing is not None:
                    logger.info(
                        f"SKIP (already completed): {config} seed={seed} seq_length={length} "
                        f"-> best_val_ppl={existing['best_val_ppl']:.4f}"
                    )
                    continue

                extra_args = [
                    f"dataset.seq_length={length}",
                    f"model.max_seq_length={length}",
                    f"training.max_epochs={max_epochs}",
                ]
                # Attention memory scales ~quadratically with seq_length at fixed batch size.
                # batch_size=32 fits at 128 but OOMs at 512 on an 8GB card — reduce it there only.
                if length == 512:
                    extra_args.append("dataset.batch_size=8")

                run_experiment(config, seed, device, extra_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pre-registered study phases.")
    parser.add_argument("--phase", choices=["variance", "full", "phase4"], default="variance")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent.parent)

    if args.device == "cuda":
        check_cuda_available()

    if args.phase == "phase4":
        baseline_stats, promoted = recompute_promoted_from_disk(args.device)
        logger.info(f"Promoted ablations recovered from disk: {promoted}")
        phase4_interaction(promoted, args.device)
    else:
        baseline_stats = phase1_variance_pilot(args.device)
        if args.phase == "full":
            promoted = phase3_screening(baseline_stats, args.device)
            phase4_interaction(promoted, args.device)

    logger.info("Study completed successfully.")
