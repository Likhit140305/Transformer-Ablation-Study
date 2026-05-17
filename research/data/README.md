# Data Directory

This directory contains the datasets used in the experiments.

## WikiText-2 ✓ DOWNLOADED AND EXTRACTED
- `wiki.train.tokens` (10.8 MB) - Training set
- `wiki.valid.tokens` (1.1 MB) - Validation set  
- `wiki.test.tokens` (1.2 MB) - Test set

## WikiText-103 ✓ DOWNLOADED AND EXTRACTED
- `wiki.train.tokens` (517.6 MB) - Training set
- `wiki.valid.tokens` (1.1 MB) - Validation set
- `wiki.test.tokens` (1.2 MB) - Test set

## Dataset Preparation Script
The datasets above were prepared using the standard preparation script. To re-download or verify datasets, you can use:
```powershell
python research/datasets/prepare_wikitext.py --output-root research/data
```

## TinyStories (optional)
For the TinyStories dataset, please refer to the original source:
https://huggingface.co/datasets/roneneldan/TinyStories

After downloading, place the dataset in a subdirectory named `tinystories`.

Note: The data paths in the configuration files are set to:
- WikiText-2: ${data_dir}/wikitext-2
- WikiText-103: ${data_dir}/wikitext-103
- TinyStories: ${data_dir}/tinystories

Where ${data_dir} is the path to this directory (or can be overridden via the `data_dir` config parameter).