"""Run the full factorial study: every condition x context length x seed.

Seeds are the outermost loop so that a partially finished study is still
balanced (every condition has the same seeds). Completed runs are detected
from schema-v3 ``results.json`` files and skipped, so the script is resumable.

    python research/experiments/run_study.py            # whole grid + budget control
    python research/experiments/run_study.py --dry-run  # list pending runs
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research" / "results"
SCHEMA_VERSION = 3

CONDITIONS = [
    "baseline",
    "abl_no_pos_enc",
    "abl_sinusoidal_pos",
    "abl_single_head",
    "abl_reduced_ffn",
    "abl_linear_ffn",
    "abl_no_residual",
    "abl_post_layernorm",
]
SEQ_LENGTHS = [128, 512]
SEEDS = [42, 1337, 2023, 7, 31415]
# Budget control: is a long-context deficit just slower convergence? Baseline only, 2x epochs.
BUDGET_CONTROL = {"name": "baseline_2x_budget", "epochs": 40, "seeds": SEEDS[:3]}
TOKENS_PER_BATCH = 4096  # batch_size * seq_length is held fixed, so steps and tokens match across lengths

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("run_study")


def completed_runs() -> set[tuple[str, int, int]]:
    done = set()
    for path in RESULTS.glob("*/L*/*/results.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("schema_version") == SCHEMA_VERSION:
            done.add((result["experiment"], result["seq_length"], result["seed"]))
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    done = completed_runs()
    grid = [(condition, length, seed, []) for seed in SEEDS for length in SEQ_LENGTHS for condition in CONDITIONS]
    grid += [
        (BUDGET_CONTROL["name"], length, seed,
         ["experiment=baseline", f"experiment_name={BUDGET_CONTROL['name']}", f"training.max_epochs={BUDGET_CONTROL['epochs']}"])
        for seed in BUDGET_CONTROL["seeds"] for length in SEQ_LENGTHS
    ]
    pending = [run for run in grid if run[:3] not in done]
    logger.info("%d runs complete, %d pending", len(done), len(pending))
    for index, (condition, length, seed, extra) in enumerate(pending, 1):
        command = [
            sys.executable, str(ROOT / "research" / "experiments" / "train.py"),
            *(extra or [f"experiment={condition}"]),
            f"seed={seed}",
            f"dataset.seq_length={length}",
            f"dataset.batch_size={TOKENS_PER_BATCH // length}",
            f"runtime.device={args.device}",
        ]
        logger.info("[%d/%d] %s L=%d seed=%d", index, len(pending), condition, length, seed)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
