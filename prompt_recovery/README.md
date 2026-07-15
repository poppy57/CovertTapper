# Grey-Box Prompt Inversion

This directory contains the implementation of a grey-box prompt inversion method. The method assumes access to intercepted target-model hidden states and reconstructs the original prompt by matching those hidden states with candidate-token forward passes.

## Method Overview

The attack combines three components:

- **Hidden-state matching:** candidate tokens are scored by the L2 distance between their target-layer hidden state and the intercepted target hidden state.
- **Cold-start search:** the first token is selected by scanning the vocabulary and keeping the nearest candidates.
- **Language-prior beam search:** later positions use a mixed candidate pool from language-prior top-k tokens and optional embedding-similarity candidates, then keep the best beam hypotheses.

The implementation is designed for reproducible experiments on Skytrax, CMS, ECHR, and synthetic Private-PII datasets.

## Directory Layout

```text
prompt_recovery/
|-- config/                  # Dataclass configuration and default YAML
|-- data/                    # CSV/JSON/JSONL dataset loaders
|-- data_cache/              # Cached evaluation CSV files
|-- methods/                 # Cold-start beam-search implementation
|-- models/                  # HuggingFace model loading and hidden-state extraction
|-- scripts/                 # Hyperparameter sweep script
|-- prepare_datasets.py      # Dataset preparation entry point
|-- run.py                   # Main grey-box inversion entry point
|-- requirements.txt         # Method-specific dependencies
`-- README.md
```

Generated `logs/` and `results/` directories are intentionally ignored by Git.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=<your_huggingface_token>
```

The default Llama models may require HuggingFace access approval. The code reads authentication only from the `HF_TOKEN` environment variable; no token is stored in the source code.

## Prepare Data

```bash
cd prompt_recovery
python prepare_datasets.py
```

This creates or verifies the CSV files in `data_cache/`:

- `skytrax_airline.csv` and sampled Skytrax subsets
- `cms_medical.csv`
- `echr_cases.csv`
- `private_pii.csv`

## Run a Single Experiment

```bash
python run.py --config config/default.yaml
```

Useful overrides:

```bash
python run.py --model llama2-7b --dataset skytrax --target_layer 16
python run.py --max_samples 5 --max_length 64 --beam_size 3 --K_prior 1000 --K_embed 500 --verbose
```

Outputs are written to:

- `logs/prompt_inversion_<timestamp>.log`
- `results/results_<model>_<timestamp>.json`

## Hyperparameter Search

```bash
bash scripts/hyperparam_search.sh
```

The script sweeps beam size and candidate-pool sizes, then writes a summary CSV under `results/hyperparam_search_<timestamp>/`.

## Main Configuration Fields

`config/default.yaml` controls the default experiment:

- `model.name`: model alias such as `llama2-7b`
- `model.target_layer`: target hidden-state layer, with `-1` meaning the final layer
- `search.K_prior`: language-prior candidate pool size
- `search.K_embed`: embedding-similarity candidate pool size
- `search.beam_size`: number of active beam hypotheses
- `data.dataset`: dataset name (`skytrax`, `cms`, `echr`, `private_pii`, `wikitext`, or `custom`)
- `experiment.output_dir`: result directory
