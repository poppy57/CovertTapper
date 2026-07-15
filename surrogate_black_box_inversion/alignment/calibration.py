"""
Calibration data:Generated (Surrogate hidden states, Target hidden states) Used forTraininglayer.

In the attack setting, the attacker can send known queries to the target system,
Model,
Model,from .
"""

import torch
import numpy as np
import logging
import os
import json
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    """Dataset"""
    h_target: torch.Tensor   # (n_samples, tgt_dim)
    h_surrogate: torch.Tensor  # (n_samples, src_dim)
    texts: List[str]
    target_layer: int
    surrogate_layer: int
    target_model: str
    surrogate_model: str

    def split(self, train_ratio: float = 0.8, seed: int = 42) -> Tuple["CalibrationData", "CalibrationData"]:
        """Training"""
        n = len(self.texts)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)
        split_idx = int(n * train_ratio)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        train_data = CalibrationData(
            h_target=self.h_target[train_idx],
            h_surrogate=self.h_surrogate[train_idx],
            texts=[self.texts[i] for i in train_idx],
            target_layer=self.target_layer,
            surrogate_layer=self.surrogate_layer,
            target_model=self.target_model,
            surrogate_model=self.surrogate_model,
        )
        val_data = CalibrationData(
            h_target=self.h_target[val_idx],
            h_surrogate=self.h_surrogate[val_idx],
            texts=[self.texts[i] for i in val_idx],
            target_layer=self.target_layer,
            surrogate_layer=self.surrogate_layer,
            target_model=self.target_model,
            surrogate_model=self.surrogate_model,
        )
        return train_data, val_data


def extract_layer_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    layer: int,
    max_length: int = 128,
    batch_size: int = 4,
    pool_method: str = "mean",
) -> torch.Tensor:
    """
    extractlayer.

    Args:
        model: Model
        tokenizer: tokenizer
        texts: input text list
        layer: layer
        max_length: sequence length
        batch_size: 
        pool_method: "mean" (mean pooling)  "last" ( padding token)

    Returns:
        (n_samples, hidden_dim) tensor
    """
    device = next(model.parameters()).device
    all_hidden = []

    for start in tqdm(range(0, len(texts), batch_size), desc=f"Extracting L{layer}"):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]

        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hs = outputs.hidden_states[layer + 1]  # [batch, seq, d]

        if pool_method == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif pool_method == "last":
            seq_lengths = attention_mask.sum(dim=1) - 1
            pooled = hs[torch.arange(hs.size(0)), seq_lengths]
        else:
            raise ValueError(f"Unknown pool method: {pool_method}")

        all_hidden.append(pooled.float().cpu())

    return torch.cat(all_hidden, dim=0)


def collect_calibration_data(
    target_model_path: str,
    surrogate_model_path: str,
    texts: List[str],
    target_layer: int,
    surrogate_layer: int,
    max_length: int = 128,
    batch_size: int = 4,
    pool_method: str = "mean",
    torch_dtype: str = "float16",
    device_map: str = "auto",
) -> CalibrationData:
    """
    Calibration data:Load model,extract.

    ModelLoad(Load model).

    Args:
        target_model_path: Model
        surrogate_model_path: Model
        texts: 
        target_layer: model layer
        surrogate_layer: Modelcorresponds tolayer( CKA layer)
        max_length: sequence length
        batch_size: 
        pool_method: pooling method
        torch_dtype: Model
        device_map: Device

    Returns:
        CalibrationData
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)

    logger.info(f"=== Calibration data ===")
    logger.info(f"  Model: {target_model_path}, layer {target_layer}")
    logger.info(f"  Model: {surrogate_model_path}, layer {surrogate_layer}")
    logger.info(f"  Sample: {len(texts)}")

    # 1. extractModel
    logger.info(f"Load model: {target_model_path}")
    tgt_tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    if tgt_tokenizer.pad_token is None:
        tgt_tokenizer.pad_token = tgt_tokenizer.eos_token

    tgt_model = AutoModelForCausalLM.from_pretrained(
        target_model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
    )
    tgt_model.eval()

    h_target = extract_layer_hidden_states(
        tgt_model, tgt_tokenizer, texts, target_layer,
        max_length=max_length, batch_size=batch_size, pool_method=pool_method
    )
    logger.info(f"Target hidden states: {h_target.shape}")

    del tgt_model, tgt_tokenizer
    torch.cuda.empty_cache()

    # 2. extractModel
    logger.info(f"Load model: {surrogate_model_path}")
    surr_tokenizer = AutoTokenizer.from_pretrained(surrogate_model_path, trust_remote_code=True)
    if surr_tokenizer.pad_token is None:
        surr_tokenizer.pad_token = surr_tokenizer.eos_token

    surr_model = AutoModelForCausalLM.from_pretrained(
        surrogate_model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
    )
    surr_model.eval()

    h_surrogate = extract_layer_hidden_states(
        surr_model, surr_tokenizer, texts, surrogate_layer,
        max_length=max_length, batch_size=batch_size, pool_method=pool_method
    )
    logger.info(f"Surrogate hidden states: {h_surrogate.shape}")

    del surr_model, surr_tokenizer
    torch.cuda.empty_cache()

    return CalibrationData(
        h_target=h_target,
        h_surrogate=h_surrogate,
        texts=texts,
        target_layer=target_layer,
        surrogate_layer=surrogate_layer,
        target_model=target_model_path.split("/")[-1],
        surrogate_model=surrogate_model_path.split("/")[-1],
    )


def save_calibration_data(calib: CalibrationData, output_dir: str):
    """Calibration datato"""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(calib.h_target, os.path.join(output_dir, "h_target.pt"))
    torch.save(calib.h_surrogate, os.path.join(output_dir, "h_surrogate.pt"))

    meta = {
        "n_samples": len(calib.texts),
        "target_dim": calib.h_target.shape[1],
        "surrogate_dim": calib.h_surrogate.shape[1],
        "target_layer": calib.target_layer,
        "surrogate_layer": calib.surrogate_layer,
        "target_model": calib.target_model,
        "surrogate_model": calib.surrogate_model,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Calibration data: {output_dir}")


def load_calibration_data(input_dir: str, texts: Optional[List[str]] = None) -> CalibrationData:
    """Load calibration data from disk."""
    h_target = torch.load(os.path.join(input_dir, "h_target.pt"), weights_only=True)
    h_surrogate = torch.load(os.path.join(input_dir, "h_surrogate.pt"), weights_only=True)

    with open(os.path.join(input_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    if texts is None:
        texts = [f"sample_{i}" for i in range(meta["n_samples"])]

    return CalibrationData(
        h_target=h_target,
        h_surrogate=h_surrogate,
        texts=texts,
        target_layer=meta["target_layer"],
        surrogate_layer=meta["surrogate_layer"],
        target_model=meta["target_model"],
        surrogate_model=meta["surrogate_model"],
    )
