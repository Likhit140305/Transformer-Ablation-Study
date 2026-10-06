# Transformer Ablation Study

A controlled, multi-seed ablation of a small decoder-only Transformer language model on
WikiText-2. It covers seven single-component ablations at two context lengths (128 and 512),
uses a matched training budget, and reports Holm-corrected statistics.
The paper is in `research/paper/`.

## Layout

```
research/
├── configs/            Hydra configs (model, dataset, training, optimizer, one file per ablation)
├── data/               WikiText-2/103 splits + SHA-256 manifest (prepare with datasets/prepare_wikitext.py)
├── datasets/           Word-level vocabulary + contiguous causal-LM chunks
├── models/             Components: attention, embedding, ffn, residual, transformer
├── experiments/
│   ├── train.py        One run: train, select best-validation epoch, evaluate on test
│   └── run_study.py    The full grid (8 conditions x 2 lengths x 5 seeds); resumable
├── analysis/
│   └── analyze_study.py  Tables, figures, Welch/Holm statistics -> research/paper/
├── tests/              Integrity tests (incl. regression test that ablations really apply)
├── results/            <experiment>/L<seq_len>/<timestamp>_seed<seed>/results.json
│   └── legacy_v1/      Pre-fix runs (ablation overrides were ignored); not used in the paper
└── paper/              IEEE LaTeX source, generated tables and figures
```

## Reproduce

Run all commands from the repository root.

```bash
pip install -r requirements.txt
```

```bash
python research/datasets/prepare_wikitext.py --output-root research/data
```

```bash
python -m pytest research/tests
```

```bash
python research/experiments/run_study.py
```

```bash
python research/analysis/analyze_study.py
```

```bash
cd research/paper && latexmk -pdf main.tex
```

A single run: `python research/experiments/train.py experiment=abl_no_pos_enc seed=42 dataset.seq_length=512 dataset.batch_size=8`.

The grid is 80 runs and takes about 10 GPU-hours on an RTX 4060 Laptop GPU with bf16 autocast.
`run_study.py` skips finished runs, so you can stop and restart it.

## Conditions

| Experiment | Change vs. baseline |
|---|---|
| `baseline` | Pre-LN, learned absolute positions, 8 heads, GELU FFN (d_ff = 4·d_model), residuals |
| `abl_no_pos_enc` | no positional encoding |
| `abl_sinusoidal_pos` | fixed sinusoidal instead of learned positions |
| `abl_single_head` | 1 head of width 256 instead of 8 × 32 (same parameters) |
| `abl_reduced_ffn` | d_ff = 256 (1×) instead of 1024 (4×) |
| `abl_linear_ffn` | identity instead of GELU (same parameters) |
| `abl_no_residual` | skip connections removed in every sub-layer |
| `abl_post_layernorm` | Post-LN instead of Pre-LN |
