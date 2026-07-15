"""Run the full surrogate black-box prompt inversion experiment.

All alignment methods use the same token-level calibration data. Partial
results are saved after each sample so interrupted runs can resume.
"""
import os, sys, json, logging, time, torch, gc
import numpy as np
from datetime import datetime
from collections import Counter

os.makedirs("results", exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alignment.align_methods import create_aligner
from methods.blackbox_search import BlackboxExtractor, BlackboxSearch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/full_experiment.log", mode="a"),
    ]
)
logger = logging.getLogger(__name__)

# -- Config --
TARGET_PATH = "meta-llama/Llama-2-7b-chat-hf"
SURROGATE_PATH = "meta-llama/Llama-2-7b-hf"
TARGET_LAYER = 31
SURROGATE_LAYER = 31

METHODS = ["A0_identity", "A1_meanvar", "A3_diagonal", "A6_full", "LRRA"]

N_CALIB = 30
N_TEST = 33
MAX_LENGTH = 512
BEAM_SIZE = 5
K_PRIOR = 500
LRRA_RANK = 64

OUTPUT_DIR = "results/full_experiment"
TOKEN_CALIB_CACHE = "results/lrra_test/token_calib_cache"


def load_skytrax_texts(data_dir, n):
    import pandas as pd
    csv_path = os.path.join(data_dir, "skytrax_airline_63.csv")
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    texts = df["content"].dropna().astype(str).tolist()
    texts = [t for t in texts if len(t.strip()) > 10]
    return texts[:n]


def compute_bleu(ref, hyp, max_n=4):
    if not hyp or not ref:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        ref_ng, hyp_ng = Counter(), Counter()
        for i in range(len(ref) - n + 1):
            ref_ng[tuple(ref[i:i+n])] += 1
        for i in range(len(hyp) - n + 1):
            hyp_ng[tuple(hyp[i:i+n])] += 1
        clipped = sum(min(hyp_ng[ng], ref_ng[ng]) for ng in hyp_ng)
        total = max(sum(hyp_ng.values()), 1)
        precisions.append(clipped / total)
    if any(p == 0 for p in precisions):
        return 0.0
    log_avg = sum(np.log(p + 1e-10) for p in precisions) / len(precisions)
    bp = min(1.0, np.exp(1 - len(ref) / max(len(hyp), 1)))
    return float(bp * np.exp(log_avg))


def collect_token_level_calibration(calib_texts, cache_dir):
    """Collect token-level calibration data, or load it from cache if available."""
    os.makedirs(cache_dir, exist_ok=True)
    h_tgt_path = os.path.join(cache_dir, "h_target_tokens.pt")
    h_surr_path = os.path.join(cache_dir, "h_surrogate_tokens.pt")

    if os.path.exists(h_tgt_path) and os.path.exists(h_surr_path):
        logger.info("Loading token-level calibration data from cache")
        h_tgt = torch.load(h_tgt_path, weights_only=True)
        h_surr = torch.load(h_surr_path, weights_only=True)
        logger.info(f"  Calibration data: {h_tgt.shape[0]} tokens, dim={h_tgt.shape[1]}")
        return h_surr, h_tgt

    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading the target model to extract token-level hidden states...")
    tgt_model = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, torch_dtype=torch.float16, device_map="auto")
    tgt_model.eval()
    tgt_dev = next(tgt_model.parameters()).device

    all_h_tgt = []
    for text in calib_texts:
        enc = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
        ids = enc["input_ids"].to(tgt_dev)
        with torch.no_grad():
            out = tgt_model(ids, output_hidden_states=True, return_dict=True)
            h = out.hidden_states[TARGET_LAYER + 1][0].float().cpu()
        all_h_tgt.append(h)

    del tgt_model; torch.cuda.empty_cache(); gc.collect()

    logger.info("Loading the surrogate model to extract token-level hidden states...")
    surr_model = AutoModelForCausalLM.from_pretrained(
        SURROGATE_PATH, torch_dtype=torch.float16, device_map="auto")
    surr_model.eval()
    surr_dev = next(surr_model.parameters()).device

    all_h_surr = []
    for text in calib_texts:
        enc = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
        ids = enc["input_ids"].to(surr_dev)
        with torch.no_grad():
            out = surr_model(ids, output_hidden_states=True, return_dict=True)
            h = out.hidden_states[SURROGATE_LAYER + 1][0].float().cpu()
        all_h_surr.append(h)

    del surr_model; torch.cuda.empty_cache(); gc.collect()

    h_tgt = torch.cat(all_h_tgt, dim=0)
    h_surr = torch.cat(all_h_surr, dim=0)
    logger.info(f"  token-level Calibration data: {h_tgt.shape[0]} tokens, dim={h_tgt.shape[1]}")
    torch.save(h_tgt, h_tgt_path)
    torch.save(h_surr, h_surr_path)
    return h_surr, h_tgt


def train_all_aligners(h_surr, h_tgt, ckpt_dir):
    """Train all alignment methods on token-level data."""
    os.makedirs(ckpt_dir, exist_ok=True)
    device = "cuda"
    d = h_surr.shape[1]

    n = h_surr.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    split = int(n * 0.85)
    train_idx, val_idx = perm[:split], perm[split:]

    h_ts = h_surr[train_idx].to(device)
    h_tt = h_tgt[train_idx].to(device)
    h_vs = h_surr[val_idx].to(device)
    h_vt = h_tgt[val_idx].to(device)

    align_info = {}
    for mid in METHODS:
        logger.info(f"Training {mid}...")
        aligner = create_aligner(mid, d, d)
        t0 = time.time()
        result = aligner.fit(h_ts, h_tt, val_surr=h_vs, val_target=h_vt)
        elapsed = time.time() - t0

        aligner.eval().to(device)
        with torch.no_grad():
            h_aligned = aligner.transform(h_vs)
            mse = float(((h_aligned - h_vt) ** 2).mean())
            cos = float((h_aligned / (h_aligned.norm(dim=1, keepdim=True) + 1e-8) *
                          h_vt / (h_vt.norm(dim=1, keepdim=True) + 1e-8)).sum(dim=1).mean())

        align_info[mid] = {
            "params": aligner.num_params, "val_mse": mse, "cos": cos,
            "train_time": round(elapsed, 2), "epochs": result.train_epochs,
        }
        logger.info(f"  {mid}: params={aligner.num_params:,}, val_mse={mse:.6f}, cos={cos:.4f}, {elapsed:.1f}s")

        torch.save(aligner.state_dict(), os.path.join(ckpt_dir, f"{mid}.pt"))
        del aligner; torch.cuda.empty_cache()

    del h_ts, h_tt, h_vs, h_vt; torch.cuda.empty_cache()
    return align_info


def _save_partial(output_dir, mid, results, align_info):
    valid = [r for r in results if not r.get("skipped")]
    partial = {
        "method": mid,
        "avg_acc": float(np.mean([r["token_accuracy"] for r in valid])) if valid else 0.0,
        "avg_bleu": float(np.mean([r["bleu"] for r in valid])) if valid else 0.0,
        "avg_oracle": float(np.mean([r["oracle_accuracy"] for r in valid])) if valid else 0.0,
        "n_completed": len(valid),
        "n_skipped": len(results) - len(valid),
        "align_info": align_info,
        "per_sample": results,
    }
    with open(os.path.join(output_dir, f"{mid}_results.json"), "w") as f:
        json.dump(partial, f, indent=2, ensure_ascii=False, default=str)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_dir = "../prompt_recovery/data_cache"

    logger.info("=" * 70)
    logger.info(": 5  x 33 Sample (token-level )")
    logger.info(f"Target L{TARGET_LAYER} -> Surrogate L{SURROGATE_LAYER}")
    logger.info("=" * 70)

    # Phase 1: Token-level Calibration data
    calib_texts = load_skytrax_texts(data_dir, N_CALIB)
    h_surr_calib, h_tgt_calib = collect_token_level_calibration(calib_texts, TOKEN_CALIB_CACHE)

    # Phase 2: train 
    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    align_info = train_all_aligners(h_surr_calib, h_tgt_calib, ckpt_dir)

    with open(os.path.join(OUTPUT_DIR, "align_info.json"), "w") as f:
        json.dump(align_info, f, indent=2, default=str)

    del h_surr_calib, h_tgt_calib; gc.collect()

    # Phase 3: extract all SampleTarget hidden states
    logger.info("\nExtracting target hidden states for the test set...")
    all_texts = load_skytrax_texts(data_dir, N_CALIB + N_TEST)
    test_texts = all_texts[N_CALIB:N_CALIB + N_TEST]

    tgt_tok = AutoTokenizer.from_pretrained(TARGET_PATH)
    if tgt_tok.pad_token is None:
        tgt_tok.pad_token = tgt_tok.eos_token
    tgt_model = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, torch_dtype=torch.float16, device_map="auto")
    tgt_model.eval()
    tgt_dev = next(tgt_model.parameters()).device

    target_samples = []
    for i, text in enumerate(test_texts):
        enc = tgt_tok(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        ids = enc["input_ids"].to(tgt_dev)
        with torch.no_grad():
            out = tgt_model(ids, output_hidden_states=True, return_dict=True)
            h = out.hidden_states[TARGET_LAYER + 1][0].float().cpu()
        target_samples.append({
            "idx": i, "text": text, "token_ids": ids[0].tolist(),
            "h_target": h, "seq_len": len(ids[0]),
        })
    logger.info(f"Test set: {len(target_samples)} records")

    del tgt_model; torch.cuda.empty_cache(); gc.collect()

    # Phase 4: load surrogate model and run inversion
    logger.info("\nLoad model...")
    surr_tok = AutoTokenizer.from_pretrained(SURROGATE_PATH)
    if surr_tok.pad_token is None:
        surr_tok.pad_token = surr_tok.eos_token
    surr_model = AutoModelForCausalLM.from_pretrained(
        SURROGATE_PATH, torch_dtype=torch.float16, device_map="auto")
    surr_model.eval()
    surr_dev = next(surr_model.parameters()).device
    d = 4096

    all_results = {}
    for mid in METHODS:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Inversion: {mid}")
        logger.info(f"{'#'*60}")

        aligner = create_aligner(mid, d, d)
        state = torch.load(os.path.join(ckpt_dir, f"{mid}.pt"), map_location="cpu", weights_only=True)
        aligner.load_state_dict(state)
        aligner.eval().to(surr_dev)

        # 
        partial_path = os.path.join(OUTPUT_DIR, f"{mid}_results.json")
        results = []
        completed_indices = set()
        if os.path.exists(partial_path):
            try:
                with open(partial_path) as pf:
                    prev = json.load(pf)
                results = prev.get("per_sample", [])
                # :Samplecompleted,skipped samplesrerun
                completed_indices = {r["idx"] for r in results if not r.get("skipped")}
                logger.info(f"  Resume: completed {len(completed_indices)}/{len(target_samples)} (rerun {len(results)-len(completed_indices)} skipped samples)")
            except Exception:
                pass

        bb = BlackboxExtractor(surr_model, surr_tok, aligner, SURROGATE_LAYER, TARGET_LAYER)
        searcher = BlackboxSearch(extractor=bb, K_prior=K_PRIOR, K_embed=0,
                                  beam_size=BEAM_SIZE, lambda_prior=0.0)

        t0 = time.time()
        for s in target_samples:
            if s["idx"] in completed_indices:
                continue

            logger.info(f"  [{mid}] Sample {s['idx']+1}/{len(target_samples)} (len={s['seq_len']})")
            h = s["h_target"].to(surr_dev)

            try:
                out = searcher.search(target_hidden_states=h, token_ids=s["token_ids"], verbose=True)
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"  [{mid}] Sample {s['idx']+1} OOM, skipped")
                torch.cuda.empty_cache(); gc.collect()
                results.append({"idx": s["idx"], "seq_len": s["seq_len"],
                                "token_accuracy": 0.0, "bleu": 0.0,
                                "oracle_accuracy": 0.0, "skipped": True})
                _save_partial(OUTPUT_DIR, mid, results, align_info[mid])
                continue

            best = out.beam_results[0] if out.beam_results else None
            acc = best.token_accuracy if best else 0.0
            bleu = compute_bleu(s["token_ids"], best.tokens) if best else 0.0
            results.append({"idx": s["idx"], "seq_len": s["seq_len"],
                            "token_accuracy": acc, "bleu": bleu,
                            "oracle_accuracy": out.oracle_accuracy})
            logger.info(f"    Acc={acc*100:.1f}% BLEU={bleu:.4f}")

            torch.cuda.empty_cache(); gc.collect()
            _save_partial(OUTPUT_DIR, mid, results, align_info[mid])

        elapsed = time.time() - t0
        valid = [r for r in results if not r.get("skipped")]
        avg_acc = float(np.mean([r["token_accuracy"] for r in valid])) if valid else 0.0
        avg_bleu = float(np.mean([r["bleu"] for r in valid])) if valid else 0.0
        avg_oracle = float(np.mean([r["oracle_accuracy"] for r in valid])) if valid else 0.0

        all_results[mid] = {
            "avg_acc": avg_acc, "avg_bleu": avg_bleu, "avg_oracle": avg_oracle,
            "elapsed": round(elapsed, 1), "n_completed": len(valid),
            "n_skipped": len(results) - len(valid),
        }
        logger.info(f"\n  [{mid}] Total: Acc={avg_acc*100:.2f}% BLEU={avg_bleu:.4f} "
                     f"Oracle={avg_oracle*100:.2f}% ({elapsed:.0f}s)")
        _save_partial(OUTPUT_DIR, mid, results, align_info[mid])

    del surr_model; torch.cuda.empty_cache(); gc.collect()

    # Phase 5: summary
    logger.info(f"\n{'='*70}")
    logger.info(f"Full experiment summary: L{TARGET_LAYER} -> L{SURROGATE_LAYER}, {N_TEST} Sample")
    logger.info(f"{'='*70}")
    logger.info(f"{'Method':<20} {'Params':>10} {'Val MSE':>10} {'Cos':>8} {'Acc':>10} {'BLEU':>10} {'Oracle':>10}")
    logger.info("-" * 80)
    for mid in METHODS:
        ai = align_info[mid]
        r = all_results[mid]
        logger.info(f"{mid:<20} {ai['params']:>10,} {ai['val_mse']:>10.6f} {ai['cos']:>8.4f} "
                     f"{r['avg_acc']*100:>9.2f}% {r['avg_bleu']:>10.4f} {r['avg_oracle']*100:>9.2f}%")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "target_layer": TARGET_LAYER, "surrogate_layer": SURROGATE_LAYER,
            "n_calib": N_CALIB, "n_test": N_TEST,
            "beam_size": BEAM_SIZE, "K_prior": K_PRIOR,
            "lrra_rank": LRRA_RANK, "calibration": "token-level",
        },
        "align_info": align_info,
        "results": all_results,
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
