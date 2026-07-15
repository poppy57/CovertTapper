"""Run cross-model CKA representation similarity analysis.

The script compares layer-wise hidden states across model pairs and writes
heatmaps, best-match curves, and summary JSON files.

Usage:
    python run_cka_analysis.py
    python run_cka_analysis.py --n_samples 100 --batch_size 2
    python run_cka_analysis.py --pairs chat_vs_base
"""

import argparse
import logging
import os
import sys
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alignment.cka_analysis import (
    CKAAnalyzer,
    CKAResult,
    save_results,
    plot_cka_heatmap,
    plot_best_match_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cka_experiment")


MODEL_REGISTRY = {
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-7b-base": "meta-llama/Llama-2-7b-hf",
    "llama2-70b-chat": "meta-llama/Llama-2-70b-chat-hf",
    "llama3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
}

MODEL_PAIRS = {
    "chat_vs_base": ("llama2-7b-chat", "llama2-7b-base"),
    "chat_vs_70b_chat": ("llama2-7b-chat", "llama2-70b-chat"),
    "chat_vs_llama3": ("llama2-7b-chat", "llama3-8b-instruct"),
    "base_vs_llama3": ("llama2-7b-base", "llama3-8b-instruct"),
}


def load_skytrax_texts(data_path: str, n_samples: int, max_length_chars: int = 1000) -> list:
    """Load text samples from the Skytrax dataset."""
    df = pd.read_csv(data_path, engine="python", on_bad_lines="skip")
    texts = df["content"].dropna().astype(str).tolist()
    texts = [t[:max_length_chars] for t in texts if len(t.strip()) > 20]
    if len(texts) < n_samples:
        logger.warning(f"Sample ({len(texts)}) is less than requested ({n_samples})")
    texts = texts[:n_samples]
    logger.info(f"Load {len(texts)} recordsSample")
    return texts


def run_pair(
    analyzer: CKAAnalyzer,
    pair_name: str,
    model_a_key: str,
    model_b_key: str,
    output_dir: str,
    batch_size: int,
    layer_stride: int,
) -> CKAResult:
    """Model CKA """
    model_a_path = MODEL_REGISTRY[model_a_key]
    model_b_path = MODEL_REGISTRY[model_b_key]

    logger.info(f"\n{'='*60}")
    logger.info(f"Model: {pair_name}")
    logger.info(f"  ModelA: {model_a_key} -> {model_a_path}")
    logger.info(f"  ModelB: {model_b_key} -> {model_b_path}")
    logger.info(f"{'='*60}")

    layers_a = None
    layers_b = None
    if layer_stride > 1:
        from transformers import AutoConfig
        cfg_a = AutoConfig.from_pretrained(model_a_path, trust_remote_code=True)
        cfg_b = AutoConfig.from_pretrained(model_b_path, trust_remote_code=True)
        layers_a = list(range(0, cfg_a.num_hidden_layers, layer_stride))
        layers_b = list(range(0, cfg_b.num_hidden_layers, layer_stride))
        if (cfg_a.num_hidden_layers - 1) not in layers_a:
            layers_a.append(cfg_a.num_hidden_layers - 1)
        if (cfg_b.num_hidden_layers - 1) not in layers_b:
            layers_b.append(cfg_b.num_hidden_layers - 1)
        logger.info(f"  layer_stride={layer_stride}: A={len(layers_a)} layers, B={len(layers_b)} layers")

    t0 = time.time()
    result = analyzer.compare_models(
        model_a_path, model_b_path,
        layers_a=layers_a, layers_b=layers_b,
        batch_size=batch_size,
    )
    elapsed = time.time() - t0
    logger.info(f"elapsed: {elapsed:.1f}s")

    pair_dir = os.path.join(output_dir, pair_name)
    os.makedirs(pair_dir, exist_ok=True)

    save_results([result], pair_dir)
    plot_cka_heatmap(result, os.path.join(pair_dir, "cka_heatmap.png"))
    plot_best_match_curve(result, os.path.join(pair_dir, "best_match_curve.png"))

    logger.info(f"5(ModelBlayer -> ModelAlayer, CKA):")
    for lb in sorted(result.best_matches.keys())[:5]:
        la, cka_val = result.best_matches[lb]
        logger.info(f"  B.layer {lb} -> A.layer {la} (CKA={cka_val:.4f})")

    return result


def generate_summary(results: list, output_dir: str):
    """Generated"""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_pairs": len(results),
        "pairs": [],
    }

    for r in results:
        avg_diag_cka = float(np.mean([
            r.cka_matrix[i, j]
            for i in range(min(r.cka_matrix.shape))
            for j in range(min(r.cka_matrix.shape))
            if i == j
        ])) if r.cka_matrix.shape[0] == r.cka_matrix.shape[1] else None

        avg_best_cka = float(np.mean([v[1] for v in r.best_matches.values()]))

        pair_info = {
            "model_a": r.model_a,
            "model_b": r.model_b,
            "matrix_shape": list(r.cka_matrix.shape),
            "avg_best_match_cka": avg_best_cka,
            "avg_diagonal_cka": avg_diag_cka,
            "max_cka": float(r.cka_matrix.max()),
            "min_cka": float(r.cka_matrix.min()),
        }
        summary["pairs"].append(pair_info)

    path = os.path.join(output_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSummary report saved: {path}")

    logger.info("\n" + "=" * 60)
    logger.info("Summary: model pairs sorted by average best-match CKA")
    logger.info("=" * 60)
    sorted_pairs = sorted(summary["pairs"], key=lambda x: x["avg_best_match_cka"], reverse=True)
    for p in sorted_pairs:
        logger.info(f"  {p['model_a']:30s} vs {p['model_b']:30s}  "
                     f"avg_best_CKA={p['avg_best_match_cka']:.4f}  max={p['max_cka']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Cross-model CKA representation similarity analysis")
    parser.add_argument("--data_path", type=str,
                        default="../prompt_recovery/data_cache/skytrax_airline.csv",
                        help="Dataset")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="number of samples used for CKA")
    parser.add_argument("--max_length", type=int, default=128,
                        help="maximum token length")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="inference batch size")
    parser.add_argument("--layer_stride", type=int, default=1,
                        help="layer sampling stride (1=all layers, 2=every other layer)")
    parser.add_argument("--torch_dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--pairs", type=str, nargs="*", default=None,
                        help=f"model pairs to compare; choices: {list(MODEL_PAIRS.keys())}")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory( results/<timestamp>)")
    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("results", f"cka_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path)
    texts = load_skytrax_texts(data_path, args.n_samples)

    analyzer = CKAAnalyzer(
        texts=texts,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
    )

    pair_names = args.pairs if args.pairs else list(MODEL_PAIRS.keys())
    all_results = []

    for pair_name in pair_names:
        if pair_name not in MODEL_PAIRS:
            logger.warning(f"Model: {pair_name},skipped")
            continue
        model_a_key, model_b_key = MODEL_PAIRS[pair_name]
        result = run_pair(
            analyzer, pair_name, model_a_key, model_b_key,
            args.output_dir, args.batch_size, args.layer_stride,
        )
        all_results.append(result)

    if all_results:
        generate_summary(all_results, args.output_dir)

    logger.info("\ncompleted!")


if __name__ == "__main__":
    main()
