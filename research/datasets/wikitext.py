"""Deterministic local WikiText data loading for causal language modelling."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WordVocabulary:
    """A vocabulary fitted exclusively on the training split."""

    token_to_id: dict[str, int]
    pad_token: str
    unk_token: str

    @property
    def size(self) -> int:
        """Return the number of tokens, including special tokens."""
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        """Return the padding token identifier."""
        return self.token_to_id[self.pad_token]

    def encode(self, tokens: Iterable[str]) -> list[int]:
        """Convert tokens to IDs, replacing unseen tokens with ``unk_token``."""
        unknown_id = self.token_to_id[self.unk_token]
        return [self.token_to_id.get(token, unknown_id) for token in tokens]

    @classmethod
    def fit(
        cls,
        train_path: Path,
        pad_token: str,
        unk_token: str,
        max_vocab_size: int | None,
        min_frequency: int,
    ) -> "WordVocabulary":
        """Fit a deterministic word vocabulary from whitespace-delimited training text."""
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        counts = Counter(train_path.read_text(encoding="utf-8").split())
        candidates = [
            (token, count) for token, count in counts.items() if count >= min_frequency
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        if max_vocab_size is not None:
            if max_vocab_size < 2:
                raise ValueError("max_vocab_size must reserve room for two special tokens")
            candidates = candidates[: max_vocab_size - 2]

        tokens = [pad_token, unk_token]
        tokens.extend(token for token, _ in candidates if token not in {pad_token, unk_token})
        return cls(token_to_id={token: index for index, token in enumerate(tokens)}, pad_token=pad_token, unk_token=unk_token)


class CausalLanguageModelDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Non-overlapping, contiguous next-token examples from one token stream."""

    def __init__(self, token_ids: list[int], seq_length: int) -> None:
        if seq_length < 1:
            raise ValueError("seq_length must be positive")
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.seq_length = seq_length
        self.num_examples = (self.tokens.numel() - 1) // seq_length
        if self.num_examples == 0:
            raise ValueError("The split does not contain enough tokens for one example")

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.seq_length
        sequence = self.tokens[start : start + self.seq_length + 1]
        return sequence[:-1], sequence[1:]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_split_checksums(data_dir: Path, dataset_dir: str, files: dict[str, str]) -> None:
    """Validate prepared files against the tracked materialization manifest."""
    manifest_path = data_dir / "DATASET_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_record = manifest["datasets"].get(dataset_dir)
    if dataset_record is None:
        raise ValueError(f"Dataset {dataset_dir!r} is absent from {manifest_path}")
    for split, filename in files.items():
        path = data_dir / dataset_dir / filename
        expected = dataset_record["records"][split]["sha256"]
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {path}: expected {expected}, received {actual}")


def load_wikitext_datasets(
    *,
    data_dir: str | Path,
    dataset_dir: str,
    files: dict[str, str],
    seq_length: int,
    pad_token: str,
    unk_token: str,
    max_vocab_size: int | None,
    min_frequency: int,
    verify_checksums: bool,
) -> tuple[CausalLanguageModelDataset, CausalLanguageModelDataset, CausalLanguageModelDataset, WordVocabulary]:
    """Load local WikiText train/validation/test splits with a training-only vocabulary."""
    root = Path(data_dir)
    split_root = root / dataset_dir
    required = {split: split_root / filename for split, filename in files.items()}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing WikiText split files: " + ", ".join(missing))
    if verify_checksums:
        _verify_split_checksums(root, dataset_dir, files)

    vocabulary = WordVocabulary.fit(
        required["train"], pad_token, unk_token, max_vocab_size, min_frequency
    )
    splits = tuple(
        CausalLanguageModelDataset(vocabulary.encode(required[name].read_text(encoding="utf-8").split()), seq_length)
        for name in ("train", "validation", "test")
    )
    return (*splits, vocabulary)
