"""Train one configuration of the controlled Transformer ablation study."""

from __future__ import annotations

import json
import logging
import math
import random
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
    if cfg.model.max_seq_length < cfg.dataset.seq_length:
        raise ValueError("model.max_seq_length must be at least dataset.seq_length")
    logger.info(
        "Loaded %d train and %d validation sequences; vocabulary size=%d",
        len(train_dataset), len(validation_dataset), vocabulary.size,
    )
    train_loader = make_loader(train_dataset, cfg, shuffle=True, seed=cfg.seed)
    validation_loader = make_loader(validation_dataset, cfg, shuffle=False, seed=cfg.seed)

    model = TransformerLM(
        vocab_size=vocabulary.size,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_ff=cfg.model.d_ff,
        max_seq_length=cfg.model.max_seq_length,
        dropout=cfg.model.dropout,
        emb_dropout=cfg.model.emb_dropout,
        activation=cfg.model.activation,
        attn_dropout=cfg.model.attn_dropout,
        ff_dropout=cfg.model.ff_dropout,
        pos_encoding=cfg.model.pos_encoding,
        pre_norm=cfg.model.pre_norm,
        use_residual=cfg.model.use_residual,
        tie_weights=cfg.model.tie_weights,
        padding_idx=vocabulary.pad_id,
    ).to(device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Model parameters: %d", total_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.max_epochs,
        eta_min=cfg.training.learning_rate * cfg.training.eta_min_ratio,
    )
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_ppl": []}
    best_val_ppl = float("inf")
    steps_completed = 0
    start_time = time.perf_counter()

    for epoch in range(cfg.training.max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_token_count = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, vocabulary.size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_val)
            optimizer.step()
            train_loss_sum += loss.item() * targets.numel()
            train_token_count += targets.numel()
            steps_completed += 1
            if cfg.training.max_steps is not None and steps_completed >= cfg.training.max_steps:
                break

        scheduler.step()
        train_loss = train_loss_sum / train_token_count
        val_loss = evaluate(model, validation_loader, criterion, device)
        val_ppl = math.exp(val_loss)
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_ppl"].append(val_ppl)
        logger.info("epoch=%d train_loss=%.4f val_loss=%.4f val_ppl=%.4f", epoch + 1, train_loss, val_loss, val_ppl)
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            if cfg.training.save_best_model:
                torch.save(model.state_dict(), output_dir / "best_model.pt")
        if cfg.training.max_steps is not None and steps_completed >= cfg.training.max_steps:
            break

    runtime_seconds = time.perf_counter() - start_time
    results = {
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
        "history": metrics,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Training complete. best_val_ppl=%.4f", best_val_ppl)
    return best_val_ppl


if __name__ == "__main__":
    train()
