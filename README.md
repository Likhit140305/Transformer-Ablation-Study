# Transformer Ablation Study

This project implements a publication-quality, statistically rigorous Transformer Ablation Study as per the ETH-Level guidelines.

## Project Structure

```
research/
├── configs/           # Hydra configuration files
├── data/              # Directory for datasets (see data/README.md)
├── models/            # Model implementations
│   ├── embedding/     # Token and positional embeddings
│   ├── attention/     # Multi-head attention
│   ├── ffn/           # Feed-forward networks
│   ├── layernorm/     # Layer normalization
│   ├── residual/      # Residual connections
│   └── transformer/   # Transformer blocks and language model
├── experiments/       # Experiment scripts (train.py files)
├── analysis/
├── paper drafts/
├── results/           # Experimental results
├── experiments/
 logs/
├── tests/
 logs/
 docs/
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Hydra
- Datasets (🤗 Hugging Face datasets) for loading WikiText
- Other dependencies: see `requirements.txt`

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the datasets:
   - See `research/data/README.md` for instructions on downloading WikiText-2 and WikiText-103
   - Place the datasets in the `research/data/` directory

### Running the Pilot Phase

As per the project guidelines, we first run the pilot phase to estimate noise and compute requirements.

#### Variance Pilot (Baseline on WikiText-2 with 4 seeds)

```bash
python research/experiments/train.py experiment=baseline_wikitext2
```
Run the above command 4 times with different seeds (you can modify the seed in the config or pass it as an argument).

#### Compute Pilot (Baseline on WikiText-103 with 1 seed)

```bash
python research/experiments/train.py experiment=baseline_wikitext103
```

### Configuration

All parameters are configured via YAML files in `research/configs/`. To override parameters, use the Hydra syntax:

```bash
python research/experiments/train.py training.max_epochs=20
```

## Next Steps

After completing the pilot phases, proceed with the architecture screening phase as outlined in the project guidelines.

## References

- Vaswani, A., et al. (2017). Attention is All You Need.
- Various ablation study methodologies from ETH Zurich, NeurIPS, ICML, ICLR, ACL, EMNLP.