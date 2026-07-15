# Prompt Inversion Experiments

This repository contains the anonymized implementation for prompt inversion experiments. The code is organized by method.

## Repository Layout

- `prompt_recovery/`: grey-box prompt inversion with hidden-state matching, language-prior candidate generation, and cold-start beam search.
- `surrogate_black_box_inversion/`: black-box prompt inversion with surrogate-model hidden states, CKA layer matching, and learned alignment modules.

Runtime outputs are intentionally excluded from the repository. New logs, checkpoints, and result JSON files are written under each method directory at run time.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=<your_huggingface_token>
```

The experiments use gated HuggingFace models in the default configuration. Set `HF_TOKEN` in your environment before running them.

## Quick Start

```bash
cd prompt_recovery
python prepare_datasets.py
python run.py --config config/default.yaml

cd ../surrogate_black_box_inversion
python run_cka_analysis.py --n_samples 50 --batch_size 1
python run_experiment.py
```

## Anonymous Release Notes

The repository excludes author names and institution-specific metadata. Large generated outputs are omitted; scripts regenerate them under `results/`.
