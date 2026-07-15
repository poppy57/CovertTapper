# Surrogate Black-Box Prompt Inversion

This directory contains the implementation of black-box prompt inversion with surrogate-model transfer. It estimates target hidden states by aligning surrogate-model hidden states into the target-model representation space, then runs beam search under the aligned hidden-state matching objective.

## Components

- `run_cka_analysis.py`: computes cross-model CKA matrices and best layer matches.
- `run_experiment.py`: trains alignment modules on token-level calibration data and evaluates black-box inversion.
- `alignment/`: CKA analysis, layer matching, calibration data collection, and alignment modules.
- `methods/blackbox_search.py`: black-box beam search using a surrogate extractor plus alignment layer.

## Quick Start

```bash
cd surrogate_black_box_inversion
python run_cka_analysis.py --n_samples 50 --batch_size 1
python run_experiment.py
```

By default, the scripts read Skytrax data from `../prompt_recovery/data_cache/`. Runtime outputs are written under `results/`, which is ignored by Git.

## Alignment Methods

The experiment includes identity, mean-variance normalization, diagonal affine, full affine, and low-rank residual alignment variants. All methods use the same token-level calibration split for fair comparison.
