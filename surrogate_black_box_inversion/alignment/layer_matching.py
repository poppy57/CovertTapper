"""
layer: CKA Modelmodel layercorresponds to.

Core attack-pipeline steps:
  CKA  -> **layer** -> alignment-method selection -> Black box

Supports:
  - CKA best match(recommended):from  CKA matrixtolayercorresponds tolayer
  - linear proportional mapping:surr_layer = round(target_layer / target_total * surr_total)
  - layer(Naive):surr_layer = target_layer
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LayerMapping:
    """layer -> layer"""
    target_layer: int
    surrogate_layer: int
    cka_score: float
    strategy: str


@dataclass
class LayerMatchResult:
    """layer"""
    target_model: str
    surrogate_model: str
    target_num_layers: int
    surrogate_num_layers: int
    strategy: str
    mappings: List[LayerMapping] = field(default_factory=list)

    def get_surrogate_layer(self, target_layer: int) -> int:
        """layer,Returncorresponds tolayer"""
        for m in self.mappings:
            if m.target_layer == target_layer:
                return m.surrogate_layer
        raise KeyError(f"No mapping found for target layer {target_layer}")

    def get_cka_score(self, target_layer: int) -> float:
        """layer,Returncorresponds to CKA """
        for m in self.mappings:
            if m.target_layer == target_layer:
                return m.cka_score
        raise KeyError(f"No mapping found for target layer {target_layer}")

    def to_dict(self) -> dict:
        return {
            "target_model": self.target_model,
            "surrogate_model": self.surrogate_model,
            "target_num_layers": self.target_num_layers,
            "surrogate_num_layers": self.surrogate_num_layers,
            "strategy": self.strategy,
            "mappings": [asdict(m) for m in self.mappings],
        }

    def summary_table(self) -> str:
        """layer"""
        lines = [
            f"Layer Matching: {self.target_model} -> {self.surrogate_model}",
            f"Strategy: {self.strategy}",
            f"Target layers: {self.target_num_layers}, Surrogate layers: {self.surrogate_num_layers}",
            "-" * 55,
            f"{'Target':>8} {'Surrogate':>10} {'CKA':>8} {'Same?':>6}",
            "-" * 55,
        ]
        for m in self.mappings:
            same = "Y" if m.target_layer == m.surrogate_layer else ""
            lines.append(f"{m.target_layer:>8} {m.surrogate_layer:>10} {m.cka_score:>8.4f} {same:>6}")
        return "\n".join(lines)


def load_cka_result(json_path: str) -> dict:
    """Load CKA  JSON """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def cka_best_match(
    cka_json_path: str,
    target_layers: Optional[List[int]] = None,
    direction: str = "target_to_surrogate",
) -> LayerMatchResult:
    """
     CKA matrixlayer.

    CKA JSON  cka_matrix[i][j] means model_a  layers_a[i] layer
     model_b  layers_b[j] layer CKA .

    In this framework:
      - model_a = target (Chat)
      - model_b = surrogate (Base / Llama-3)
      -  target layer i, surrogate layer j such that cka_matrix[i][j] 

    Args:
        cka_json_path: CKA result JSON file path
        target_layers: layer(None = all)
        direction: "target_to_surrogate"  "surrogate_to_target"

    Returns:
        LayerMatchResult
    """
    data = load_cka_result(cka_json_path)
    cka_matrix = np.array(data["cka_matrix"])  # shape: (n_layers_a, n_layers_b)
    layers_a = data["layers_a"]  # target layers
    layers_b = data["layers_b"]  # surrogate layers
    model_a = data["model_a"]
    model_b = data["model_b"]

    if target_layers is None:
        target_layers = layers_a

    mappings = []
    for tl in target_layers:
        if tl not in layers_a:
            logger.warning(f"Target layer {tl} not in CKA result layers_a, skipping")
            continue
        i = layers_a.index(tl)
        row = cka_matrix[i]
        best_j = int(np.argmax(row))
        best_cka = float(row[best_j])
        surrogate_layer = layers_b[best_j]
        mappings.append(LayerMapping(
            target_layer=tl,
            surrogate_layer=surrogate_layer,
            cka_score=best_cka,
            strategy="cka_best",
        ))

    result = LayerMatchResult(
        target_model=model_a,
        surrogate_model=model_b,
        target_num_layers=len(layers_a),
        surrogate_num_layers=len(layers_b),
        strategy="cka_best",
        mappings=mappings,
    )
    return result


def proportional_match(
    target_num_layers: int,
    surrogate_num_layers: int,
    target_layers: Optional[List[int]] = None,
    target_model: str = "",
    surrogate_model: str = "",
    cka_json_path: Optional[str] = None,
) -> LayerMatchResult:
    """
    linear proportional mapping:surr_layer = round(target_layer / target_total * surr_total)

    If provided cka_json_path,corresponds to CKA Reference.
    """
    if target_layers is None:
        target_layers = list(range(target_num_layers))

    cka_matrix = None
    layers_a = layers_b = None
    if cka_json_path and os.path.exists(cka_json_path):
        data = load_cka_result(cka_json_path)
        cka_matrix = np.array(data["cka_matrix"])
        layers_a = data["layers_a"]
        layers_b = data["layers_b"]

    mappings = []
    for tl in target_layers:
        sl = round(tl / max(target_num_layers - 1, 1) * (surrogate_num_layers - 1))
        sl = min(sl, surrogate_num_layers - 1)

        cka_score = 0.0
        if cka_matrix is not None and tl in layers_a and sl in layers_b:
            i = layers_a.index(tl)
            j = layers_b.index(sl)
            cka_score = float(cka_matrix[i][j])

        mappings.append(LayerMapping(
            target_layer=tl,
            surrogate_layer=sl,
            cka_score=cka_score,
            strategy="proportional",
        ))

    return LayerMatchResult(
        target_model=target_model,
        surrogate_model=surrogate_model,
        target_num_layers=target_num_layers,
        surrogate_num_layers=surrogate_num_layers,
        strategy="proportional",
        mappings=mappings,
    )


def same_layer_match(
    target_num_layers: int,
    surrogate_num_layers: int,
    target_layers: Optional[List[int]] = None,
    target_model: str = "",
    surrogate_model: str = "",
    cka_json_path: Optional[str] = None,
) -> LayerMatchResult:
    """
    Naive :layer(layer clip).
    """
    if target_layers is None:
        target_layers = list(range(target_num_layers))

    cka_matrix = None
    layers_a = layers_b = None
    if cka_json_path and os.path.exists(cka_json_path):
        data = load_cka_result(cka_json_path)
        cka_matrix = np.array(data["cka_matrix"])
        layers_a = data["layers_a"]
        layers_b = data["layers_b"]

    mappings = []
    for tl in target_layers:
        sl = min(tl, surrogate_num_layers - 1)

        cka_score = 0.0
        if cka_matrix is not None and tl in layers_a and sl in layers_b:
            i = layers_a.index(tl)
            j = layers_b.index(sl)
            cka_score = float(cka_matrix[i][j])

        mappings.append(LayerMapping(
            target_layer=tl,
            surrogate_layer=sl,
            cka_score=cka_score,
            strategy="same_layer",
        ))

    return LayerMatchResult(
        target_model=target_model,
        surrogate_model=surrogate_model,
        target_num_layers=target_num_layers,
        surrogate_num_layers=surrogate_num_layers,
        strategy="same_layer",
        mappings=mappings,
    )


def compare_strategies(
    cka_json_path: str,
    target_layers: Optional[List[int]] = None,
) -> Dict[str, LayerMatchResult]:
    """
    layer.

    Returns:
        {"cka_best": ..., "proportional": ..., "same_layer": ...}
    """
    data = load_cka_result(cka_json_path)
    model_a = data["model_a"]
    model_b = data["model_b"]
    n_a = len(data["layers_a"])
    n_b = len(data["layers_b"])

    cka_result = cka_best_match(cka_json_path, target_layers)
    prop_result = proportional_match(
        n_a, n_b, target_layers, model_a, model_b, cka_json_path
    )
    same_result = same_layer_match(
        n_a, n_b, target_layers, model_a, model_b, cka_json_path
    )

    return {
        "cka_best": cka_result,
        "proportional": prop_result,
        "same_layer": same_result,
    }


def save_layer_mapping(result: LayerMatchResult, output_path: str):
    """layerto JSON"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f"layerResults saved: {output_path}")


def load_layer_mapping(json_path: str) -> LayerMatchResult:
    """from  JSON Loadlayer"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mappings = [
        LayerMapping(**m) for m in data["mappings"]
    ]
    return LayerMatchResult(
        target_model=data["target_model"],
        surrogate_model=data["surrogate_model"],
        target_num_layers=data["target_num_layers"],
        surrogate_num_layers=data["surrogate_num_layers"],
        strategy=data["strategy"],
        mappings=mappings,
    )
