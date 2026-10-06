# Configs

Hydra config tree composed by `config.yaml`. Each `experiment/*.yaml` declares exactly one
`model_overrides` change relative to `model/transformer_base.yaml`; `train.py` merges it in
(`resolved_model_config`) and `tests/` verifies every ablation changes only its declared field.
