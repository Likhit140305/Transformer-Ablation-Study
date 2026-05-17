"""Materialize the WikiText v1 word-level splits used by the experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import datasets
from datasets import load_dataset


DATASETS = {
    "wikitext-2": "wikitext-2-v1",
    "wikitext-103": "wikitext-103-v1",
}
SPLIT_FILENAMES = {
    "train": "wiki.train.tokens",
    "validation": "wiki.valid.tokens",
    "test": "wiki.test.tokens",
}
SOURCE = "Salesforce/wikitext"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_split(texts: list[str], destination: Path) -> None:
    """Write examples as newline-delimited text, preserving empty examples."""
    destination.write_text("\n".join(texts) + "\n", encoding="utf-8")


def prepare_dataset(output_root: Path, name: str, configuration: str) -> dict[str, Any]:
    """Download one WikiText configuration and record its materialized splits."""
    dataset = load_dataset(SOURCE, configuration)
    destination = output_root / name
    destination.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for split, filename in SPLIT_FILENAMES.items():
        path = destination / filename
        write_split(dataset[split]["text"], path)
        records[split] = {
            "file": str(path.relative_to(output_root)),
            "rows": len(dataset[split]),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    return {"configuration": configuration, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("research/data"))
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    args = parser.parse_args()

    selected = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source": f"https://huggingface.co/datasets/{SOURCE}",
        "variant": "word-level v1 (non-raw)",
        "datasets_library_version": datasets.__version__,
        "datasets": {},
    }
    for name, configuration in selected.items():
        manifest["datasets"][name] = prepare_dataset(args.output_root, name, configuration)
    (args.output_root / "DATASET_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
