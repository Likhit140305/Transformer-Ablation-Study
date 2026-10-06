"""Train one condition of the controlled Transformer ablation study.

Selects the checkpoint with the best validation perplexity and reports its
test perplexity. Writes ``results.json`` (schema v3) into the Hydra run dir.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.datasets.wikitext import load_wikitext_datasets
from research.models.transformer.transformer import TransformerLM

SCHEMA_VERSION = 3
AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp32": None}
logger = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool, benchmark: bool) -> None:
    """Set all random states before model or loader creation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def make_loader(dataset: Dataset, cfg: DictConfig, shuffle: bool, seed: int) -> DataLoader:
    """Seed-controlled loader; evaluation order is always fixed."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=shuffle,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        generator=generator,
    )


def lm_loss(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
    """Mean next-token cross-entropy; the loss itself is always computed in fp32."""
    with torch.autocast(inputs.device.type, dtype=dtype, enabled=dtype is not None):
        logits = model(inputs)
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, dtype: torch.dtype | None) -> float:
    """Token-weighted mean cross-entropy (nats/token) over a whole split."""
    model.eval()
    loss_sum, token_count = 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        loss_sum += lm_loss(model, inputs, targets, dtype).item() * targets.numel()
        token_count += targets.numel()
    return loss_sum / token_count


def resolved_model_config(cfg: DictConfig) -> DictConfig:
    """Merge the experiment's declared overrides into the base model config."""
    return OmegaConf.merge(cfg.model, cfg.experiment.get("model_overrides") or {})


def run_signature(cfg: DictConfig, model_cfg: DictConfig) -> str:
    """Stable condition identifier; excludes seed and output paths."""
    payload = {
        key: OmegaConf.to_container(node, resolve=True)
        for key, node in {
            "model": model_cfg,
            "dataset": cfg.dataset,
            "training": cfg.training,
            "optimizer": cfg.optimizer,
        }.items()
    }
    payload["experiment"] = cfg.experiment_name
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def lr_multiplier(step: int, *, warmup_steps: int, total_steps: int, scheduler: str, eta_min_ratio: float) -> float:
    """Linear warmup followed by constant, linear, or cosine decay to ``eta_min_ratio``."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if scheduler == "constant":
        return 1.0
    progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
    if scheduler == "linear":
        return 1.0 - (1.0 - eta_min_ratio) * progress
    if scheduler == "cosine":
        return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unsupported scheduler: {scheduler}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> float:
    seed_everything(cfg.seed, cfg.training.deterministic, cfg.training.benchmark)
    if cfg.runtime.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required by the configuration but unavailable; override runtime.device=cpu explicitly.")
    device = torch.device(cfg.runtime.device)
    dtype = AUTOCAST_DTYPES[cfg.training.precision]
    model_cfg = resolved_model_config(cfg)
    logger.info("Starting %s | seed=%d | seq_length=%d | model=%s", cfg.experiment_name, cfg.seed,
                cfg.dataset.seq_length, OmegaConf.to_container(model_cfg, resolve=True))

    train_ds, val_ds, test_ds, vocabulary = load_wikitext_datasets(
        data_dir=cfg.paths.data_dir,
        dataset_dir=cfg.dataset.dataset_dir,
        files={"train": cfg.dataset.train_file, "validation": cfg.dataset.validation_file, "test": cfg.dataset.test_file},
        seq_length=cfg.dataset.seq_length,
        pad_token=cfg.dataset.pad_token,
        unk_token=cfg.dataset.unk_token,
        max_vocab_size=cfg.dataset.max_vocab_size,
        min_frequency=cfg.dataset.min_frequency,
        verify_checksums=cfg.dataset.verify_checksums,
    )
    logger.info("Sequences: train=%d val=%d test=%d | vocab=%d", len(train_ds), len(val_ds), len(test_ds), vocabulary.size)
    train_loader = make_loader(train_ds, cfg, shuffle=True, seed=cfg.seed)
    val_loader = make_loader(val_ds, cfg, shuffle=False, seed=cfg.seed)
    test_loader = make_loader(test_ds, cfg, shuffle=False, seed=cfg.seed)

    model_kwargs = {k: v for k, v in OmegaConf.to_container(model_cfg, resolve=True).items() if k != "_target_"}
    model = TransformerLM(vocab_size=vocabulary.size, padding_idx=vocabulary.pad_id, **model_kwargs).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", total_params)

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    accumulate = cfg.training.accumulate_grad_batches
    total_steps = cfg.training.max_steps or math.ceil(len(train_loader) / accumulate) * cfg.training.max_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_multiplier(
            step,
            warmup_steps=min(cfg.training.warmup_steps, total_steps),
            total_steps=total_steps,
            scheduler=cfg.training.scheduler,
            eta_min_ratio=cfg.training.eta_min_ratio,
        ),
    )

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_ppl": [], "learning_rate": [], "grad_norm": []}
    best_val_loss, best_epoch, best_state = float("inf"), -1, None
    steps_completed = 0
    diverged = False
    start_time = time.perf_counter()

    for epoch in range(cfg.training.max_epochs):
        model.train()
        loss_sum, token_count, grad_norms = torch.zeros((), device=device), 0, []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            loss = lm_loss(model, inputs, targets, dtype)
            (loss / accumulate).backward()
            loss_sum += loss.detach() * targets.numel()  # no host sync per step
            token_count += targets.numel()
            if (batch_index + 1) % accumulate and batch_index + 1 != len(train_loader):
                continue
            grad_norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_val))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            steps_completed += 1
            if steps_completed == total_steps:
                break

        train_loss = loss_sum.item() / token_count
        val_loss = evaluate(model, val_loader, device, dtype)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(math.exp(val_loss))
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        history["grad_norm"].append(torch.stack(grad_norms).median().item())
        logger.info("epoch=%d train_loss=%.4f val_loss=%.4f val_ppl=%.2f", epoch + 1, train_loss, val_loss, math.exp(val_loss))
        if not math.isfinite(train_loss):
            diverged = True
            logger.warning("Training diverged at epoch %d", epoch + 1)
            break
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        if steps_completed == total_steps:
            break

    runtime_seconds = time.perf_counter() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)
        if cfg.training.save_best_model:
            torch.save(best_state, output_dir / "best_model.pt")
    test_loss = evaluate(model, test_loader, device, dtype) if best_state is not None else float("nan")

    results = {
        "schema_version": SCHEMA_VERSION,
        "experiment": cfg.experiment_name,
        "seed": cfg.seed,
        "seq_length": cfg.dataset.seq_length,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_ppl": math.exp(best_val_loss),
        "test_loss": test_loss,
        "test_ppl": math.exp(test_loss),
        "diverged": diverged,
        "runtime_seconds": runtime_seconds,
        "total_params": total_params,
        "vocab_size": vocabulary.size,
        "train_examples": len(train_ds),
        "dataset_name": cfg.dataset.name,
        "steps_completed": steps_completed,
        "condition_signature": run_signature(cfg, model_cfg),
        "git_commit": git_commit(),
        "resolved_model": model_kwargs,
        "history": history,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Done: best_val_ppl=%.3f (epoch %d) test_ppl=%.3f", math.exp(best_val_loss), best_epoch, math.exp(test_loss))
    return math.exp(best_val_loss)


if __name__ == "__main__":
    train()
