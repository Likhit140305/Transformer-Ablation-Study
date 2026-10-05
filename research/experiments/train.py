"""Train one configuration of the controlled Transformer ablation study."""

from __future__ import annotations

import json
import logging
import math
import random
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.datasets.wikitext import load_wikitext_datasets
from research.models.transformer.transformer import TransformerLM

logger = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool, benchmark: bool) -> None:
    """Set all experiment-level random states before model or loader creation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def make_loader(dataset: Dataset, cfg: DictConfig, shuffle: bool, seed: int) -> DataLoader:
    """Create a seed-controlled loader; validation order always remains fixed."""
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


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """Return token-weighted validation cross-entropy."""
    model.eval()
    loss_sum = 0.0
    token_count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss_sum += loss.item() * targets.numel()
            token_count += targets.numel()
    return loss_sum / token_count


def resolved_model_config(cfg: DictConfig) -> DictConfig:
    """Merge declared experiment overrides into the actual model condition."""
    return OmegaConf.merge(cfg.model, cfg.experiment.get("model_overrides", {}))


def run_signature(cfg: DictConfig, model_cfg: DictConfig) -> str:
    """Stable condition identifier, deliberately excluding seed and output paths."""
    payload = {
        "experiment": cfg.experiment_name,
        "experiment_metadata": OmegaConf.to_container(cfg.experiment, resolve=True),
        "model": OmegaConf.to_container(model_cfg, resolve=True),
        "dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
        "training": OmegaConf.to_container(cfg.training, resolve=True),
        "optimizer": OmegaConf.to_container(cfg.optimizer, resolve=True),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def git_commit() -> str | None:
    """Record source provenance without making Git a run-time requirement."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig, total_steps: int):
    """Implement configured warmup plus constant, linear, or cosine decay."""
    warmup_steps = min(int(cfg.training.warmup_steps), total_steps)
    eta_min_ratio = float(cfg.training.eta_min_ratio)
    name = str(cfg.training.scheduler).lower()
    if name not in {"cosine", "linear", "constant"}:
        raise ValueError(f"Unsupported scheduler: {name}")

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        if name == "constant":
            return 1.0
        progress = min(max((step - warmup_steps) / max(1, total_steps - warmup_steps), 0.0), 1.0)
        if name == "linear":
            return 1.0 - (1.0 - eta_min_ratio) * progress
        return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> float:
    """Train the configured model and return its best validation perplexity."""
    seed_everything(cfg.seed, cfg.training.deterministic, cfg.training.benchmark)
    if cfg.runtime.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was required by the experiment configuration but is unavailable. "
            "Install a CUDA-enabled PyTorch build or explicitly override runtime.device=cpu."
        )
    device = torch.device(cfg.runtime.device)
    logger.info("Starting %s | seed=%d | device=%s", cfg.experiment_name, cfg.seed, device)
    model_cfg = resolved_model_config(cfg)

    split_files = {
        "train": cfg.dataset.train_file,
        "validation": cfg.dataset.validation_file,
        "test": cfg.dataset.test_file,
    }
    train_dataset, validation_dataset, vocabulary = load_wikitext_datasets(
        data_dir=cfg.paths.data_dir,
        dataset_dir=cfg.dataset.dataset_dir,
        files=split_files,
        seq_length=cfg.dataset.seq_length,
        pad_token=cfg.dataset.pad_token,
        unk_token=cfg.dataset.unk_token,
        max_vocab_size=cfg.dataset.max_vocab_size,
        min_frequency=cfg.dataset.min_frequency,
        verify_checksums=cfg.dataset.verify_checksums,
    )
    if model_cfg.max_seq_length < cfg.dataset.seq_length:
        raise ValueError("model.max_seq_length must be at least dataset.seq_length")
    logger.info(
        "Loaded %d train and %d validation sequences; vocabulary size=%d",
        len(train_dataset), len(validation_dataset), vocabulary.size,
    )
    train_loader = make_loader(train_dataset, cfg, shuffle=True, seed=cfg.seed)
    validation_loader = make_loader(validation_dataset, cfg, shuffle=False, seed=cfg.seed)

    model = TransformerLM(
        vocab_size=vocabulary.size,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        d_ff=model_cfg.d_ff,
        max_seq_length=model_cfg.max_seq_length,
        dropout=model_cfg.dropout,
        emb_dropout=model_cfg.emb_dropout,
        activation=model_cfg.activation,
        attn_dropout=model_cfg.attn_dropout,
        ff_dropout=model_cfg.ff_dropout,
        pos_encoding=model_cfg.pos_encoding,
        pre_norm=model_cfg.pre_norm,
        use_residual=model_cfg.use_residual,
        tie_weights=model_cfg.tie_weights,
        padding_idx=vocabulary.pad_id,
    ).to(device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Model parameters: %d", total_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    steps_per_epoch = math.ceil(len(train_loader) / cfg.training.accumulate_grad_batches)
    total_steps = cfg.training.max_steps or steps_per_epoch * cfg.training.max_epochs
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_ppl": [], "learning_rate": []}
    best_val_ppl = float("inf")
    steps_completed = 0
    start_time = time.perf_counter()

    for epoch in range(cfg.training.max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_token_count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, vocabulary.size), targets.reshape(-1))
            (loss / cfg.training.accumulate_grad_batches).backward()
            train_loss_sum += loss.item() * targets.numel()
            train_token_count += targets.numel()
            is_update = (batch_index + 1) % cfg.training.accumulate_grad_batches == 0 or batch_index + 1 == len(train_loader)
            if not is_update:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_val)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            steps_completed += 1
            if cfg.training.max_steps is not None and steps_completed >= cfg.training.max_steps:
                break

        train_loss = train_loss_sum / train_token_count
        val_loss = evaluate(model, validation_loader, criterion, device)
        val_ppl = math.exp(val_loss)
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_ppl"].append(val_ppl)
        metrics["learning_rate"].append(optimizer.param_groups[0]["lr"])
        logger.info("epoch=%d train_loss=%.4f val_loss=%.4f val_ppl=%.4f", epoch + 1, train_loss, val_loss, val_ppl)
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            if cfg.training.save_best_model:
                torch.save(model.state_dict(), output_dir / "best_model.pt")
        if cfg.training.max_steps is not None and steps_completed >= cfg.training.max_steps:
            break

    runtime_seconds = time.perf_counter() - start_time
    if cfg.training.save_final_model:
        torch.save(model.state_dict(), output_dir / "final_model.pt")
    results = {
        "schema_version": 2,
        "experiment": cfg.experiment_name,
        "seed": cfg.seed,
        "best_val_ppl": best_val_ppl,
        "runtime_seconds": runtime_seconds,
        "total_params": total_params,
        "vocab_size": vocabulary.size,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "data_source": "local_verified_wikitext",
        "dataset_name": cfg.dataset.name,
        "steps_completed": steps_completed,
        "condition_signature": run_signature(cfg, model_cfg),
        "git_commit": git_commit(),
        "resolved_model": OmegaConf.to_container(model_cfg, resolve=True),
        "history": metrics,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Training complete. best_val_ppl=%.4f", best_val_ppl)
    return best_val_ppl


if __name__ == "__main__":
    train()
