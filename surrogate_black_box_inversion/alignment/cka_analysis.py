"""
CKA (Centered Kernel Alignment) representation similarity analysis

Used formodel layer,
Generated layerxlayer  CKA ,layercorresponds to.

Reference: Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019.
"""

import torch
import numpy as np
import logging
import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear CKA.

    Args:
        X: (n, p) ModelAmatrix,nSample,pfeatures
        Y: (n, q) ModelBmatrix,nSample,qfeatures

    Returns:
        CKA similarity in the range [0, 1]
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    hsic_xy = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord="fro")
    hsic_yy = np.linalg.norm(Y.T @ Y, ord="fro")

    denom = hsic_xx * hsic_yy
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def linear_cka_batch(
    representations_a: Dict[int, np.ndarray],
    representations_b: Dict[int, np.ndarray],
) -> np.ndarray:
    """
    layer CKA matrix.

    Args:
        representations_a: {layer_idx: (n, d)} ModelAlayer
        representations_b: {layer_idx: (n, d)} ModelBlayer

    Returns:
        CKA matrix (num_layers_a, num_layers_b)
    """
    layers_a = sorted(representations_a.keys())
    layers_b = sorted(representations_b.keys())
    cka_matrix = np.zeros((len(layers_a), len(layers_b)))

    for i, la in enumerate(tqdm(layers_a, desc="Computing CKA")):
        for j, lb in enumerate(layers_b):
            cka_matrix[i, j] = linear_cka(
                representations_a[la], representations_b[lb]
            )

    return cka_matrix


@dataclass
class ModelSpec:
    """Model"""
    name: str
    path: str
    num_layers: int = 0
    hidden_size: int = 0


@dataclass
class CKAResult:
    """CKA analysis result"""
    model_a: str
    model_b: str
    cka_matrix: np.ndarray
    layers_a: List[int] = field(default_factory=list)
    layers_b: List[int] = field(default_factory=list)
    best_matches: Dict[int, Tuple[int, float]] = field(default_factory=dict)
    n_samples: int = 0

    def find_best_matches(self):
        """ModelBlayertoModelA CKA layer"""
        self.best_matches = {}
        for j, lb in enumerate(self.layers_b):
            best_i = int(np.argmax(self.cka_matrix[:, j]))
            self.best_matches[lb] = (self.layers_a[best_i], float(self.cka_matrix[best_i, j]))

    def to_dict(self) -> dict:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "cka_matrix": self.cka_matrix.tolist(),
            "layers_a": self.layers_a,
            "layers_b": self.layers_b,
            "best_matches": {str(k): [v[0], v[1]] for k, v in self.best_matches.items()},
            "n_samples": self.n_samples,
        }


class CKAAnalyzer:
    """
    Model CKA representation similarity analysis.

    Workflow:
    1. Modelextract(mean-pool over seq_len)
    2. layer CKA
    3. layer
    """

    def __init__(
        self,
        texts: List[str],
        max_length: int = 128,
        torch_dtype: str = "float16",
        device_map: str = "auto",
    ):
        self.texts = texts
        self.max_length = max_length
        self.dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.torch_dtype = self.dtype_map.get(torch_dtype, torch.float16)
        self.device_map = device_map

    @torch.no_grad()
    def extract_representations(
        self,
        model_path: str,
        layer_indices: Optional[List[int]] = None,
        batch_size: int = 4,
    ) -> Tuple[Dict[int, np.ndarray], ModelSpec]:
        """
        extractmodel layer(mean-pool over sequence).

        Model,.

        Args:
            model_path: HuggingFace Model
            layer_indices: extractlayer(None=alllayer)
            batch_size: batch size

        Returns:
            (representations, model_spec)
            representations: {layer_idx: np.ndarray of shape (n_samples, hidden_dim)}
        """
        logger.info(f"Load model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        )
        model.eval()

        num_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        spec = ModelSpec(
            name=model_path.split("/")[-1],
            path=model_path,
            num_layers=num_layers,
            hidden_size=hidden_size,
        )

        if layer_indices is None:
            layer_indices = list(range(num_layers))
        layer_set = set(layer_indices)

        logger.info(f"  layer={num_layers}, dimension={hidden_size}, extract {len(layer_indices)} layer")

        representations = {l: [] for l in layer_indices}
        n_samples = len(self.texts)

        for start in tqdm(range(0, n_samples, batch_size), desc=f"Extracting {spec.name}"):
            end = min(start + batch_size, n_samples)
            batch_texts = self.texts[start:end]

            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            # outputs.hidden_states: tuple of (num_layers+1) tensors
            # [0] = embedding output, [i] = layer i output
            for layer_idx in layer_indices:
                hs = outputs.hidden_states[layer_idx + 1]  # [batch, seq, d]
                mask = attention_mask.unsqueeze(-1).float()  # [batch, seq, 1]
                pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [batch, d]
                representations[layer_idx].append(pooled.float().cpu().numpy())

        for layer_idx in layer_indices:
            representations[layer_idx] = np.concatenate(representations[layer_idx], axis=0)

        del model
        torch.cuda.empty_cache()
        logger.info(f"Model {spec.name} extractcompleted,")

        return representations, spec

    def compare_models(
        self,
        model_a_path: str,
        model_b_path: str,
        layers_a: Optional[List[int]] = None,
        layers_b: Optional[List[int]] = None,
        batch_size: int = 4,
    ) -> CKAResult:
        """
        Model CKA matrix.

        Args:
            model_a_path: ModelA
            model_b_path: ModelB
            layers_a: ModelAlayer(None=all)
            layers_b: ModelBlayer(None=all)
            batch_size: batch size

        Returns:
            CKAResult  CKA matrix
        """
        repr_a, spec_a = self.extract_representations(model_a_path, layers_a, batch_size)
        repr_b, spec_b = self.extract_representations(model_b_path, layers_b, batch_size)

        logger.info(f"Computing CKA: {spec_a.name} ({len(repr_a)} layers) vs {spec_b.name} ({len(repr_b)} layers)")
        cka_matrix = linear_cka_batch(repr_a, repr_b)

        result = CKAResult(
            model_a=spec_a.name,
            model_b=spec_b.name,
            cka_matrix=cka_matrix,
            layers_a=sorted(repr_a.keys()),
            layers_b=sorted(repr_b.keys()),
            n_samples=len(self.texts),
        )
        result.find_best_matches()

        return result


def save_results(results: List[CKAResult], output_dir: str):
    """ CKA analysis resultto JSON"""
    os.makedirs(output_dir, exist_ok=True)
    for r in results:
        fname = f"cka_{r.model_a}_vs_{r.model_b}.json"
        path = os.path.join(output_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(r.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved: {path}")


def plot_cka_heatmap(result: CKAResult, output_path: str):
    """Plot CKA heatmap"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(result.cka_matrix, cmap="magma", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Linear CKA")

    n_a = len(result.layers_a)
    n_b = len(result.layers_b)
    tick_step_a = max(1, n_a // 10)
    tick_step_b = max(1, n_b // 10)
    ax.set_xticks(range(0, n_b, tick_step_b))
    ax.set_xticklabels([result.layers_b[i] for i in range(0, n_b, tick_step_b)])
    ax.set_yticks(range(0, n_a, tick_step_a))
    ax.set_yticklabels([result.layers_a[i] for i in range(0, n_a, tick_step_a)])

    ax.set_xlabel(f"{result.model_b} Layer", fontsize=12)
    ax.set_ylabel(f"{result.model_a} Layer", fontsize=12)
    ax.set_title(
        f"CKA: {result.model_a} vs {result.model_b}\n(n={result.n_samples} samples)",
        fontsize=13,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap saved: {output_path}")


def plot_best_match_curve(result: CKAResult, output_path: str):
    """layer"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers_b = sorted(result.best_matches.keys())
    matched_a = [result.best_matches[lb][0] for lb in layers_b]
    cka_values = [result.best_matches[lb][1] for lb in layers_b]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(layers_b, matched_a, "o-", color="steelblue", markersize=4)
    max_a = max(result.layers_a)
    max_b = max(result.layers_b)
    ax1.plot(
        [0, max_b], [0, max_a],
        "--", color="gray", alpha=0.5, label="Linear proportion",
    )
    ax1.set_xlabel(f"{result.model_b} Layer", fontsize=11)
    ax1.set_ylabel(f"Best Matching {result.model_a} Layer", fontsize=11)
    ax1.set_title("Best Layer Correspondence", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(layers_b, cka_values, color="coral", alpha=0.8, width=0.8)
    ax2.set_xlabel(f"{result.model_b} Layer", fontsize=11)
    ax2.set_ylabel("CKA Similarity", fontsize=11)
    ax2.set_title("Best Match CKA Score per Layer", fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"{result.model_a} vs {result.model_b}", fontsize=14, y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Best-match curve saved: {output_path}")
