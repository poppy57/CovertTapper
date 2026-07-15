"""
Black box:Model+layer.

Difference from the grey-box search:
  Grey box: candidate token -> Model -> Target hidden states -> compare distance with intercepted target h*
  Black box: candidate token -> Model -> Surrogate hidden states -> alignment transform -> compare distance with intercepted target h*

 BlackboxExtractor Model+layer,and exposes the same interface as HiddenStateExtractor,
such that ColdstartSearch reuse.
"""

import torch
import logging
import time
import json
import os
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from alignment.align_methods import BaseAligner

logger = logging.getLogger(__name__)


@dataclass
class PrefixCache:
    """Prefix KV cache and related metadata()"""
    prefix_ids: List[int]
    past_key_values: Any
    prior_candidates: List[Tuple[int, float]]
    prior_log_probs: Dict[int, float]


@dataclass
class BeamState:
    prefix_ids: List[int]
    score: float
    distances: List[float]
    log_probs: List[float]

    def copy(self) -> "BeamState":
        return BeamState(
            prefix_ids=self.prefix_ids.copy(),
            score=self.score,
            distances=self.distances.copy(),
            log_probs=self.log_probs.copy(),
        )

    def get_prefix_key(self) -> tuple:
        return tuple(self.prefix_ids)


@dataclass
class BeamResult:
    rank: int
    tokens: List[int]
    text: str
    score: float
    avg_distance: float
    avg_log_prob: float
    token_accuracy: float
    token_details: List[Dict] = field(default_factory=list)


@dataclass
class SearchOutput:
    beam_results: List[BeamResult]
    target_tokens: List[int]
    target_text: str
    best_accuracy: float
    oracle_accuracy: float


class BlackboxExtractor:
    """
    Black boxHidden-state extractor.

    Model + layerModel.
    The public interface matches HiddenStateExtractor.
    """

    def __init__(
        self,
        surrogate_model: AutoModelForCausalLM,
        surrogate_tokenizer: AutoTokenizer,
        aligner: BaseAligner,
        surrogate_layer: int,
        target_layer: int,
    ):
        self.model = surrogate_model
        self.tokenizer = surrogate_tokenizer
        self.aligner = aligner
        self.surrogate_layer = surrogate_layer
        self.target_layer = target_layer
        self.device = next(surrogate_model.parameters()).device
        self.num_layers = surrogate_model.config.num_hidden_layers
        self.lm_head = surrogate_model.lm_head.weight.detach()
        self.vocab_size = len(surrogate_tokenizer)

        self.aligner.eval()
        self.aligner.to(self.device)

    @torch.no_grad()
    def extract_hidden_states_aligned(self, token_ids: List[int]) -> torch.Tensor:
        """
        extractmodel layer.
        Return [seq_len, tgt_hidden_size].
        """
        input_ids = torch.tensor([token_ids], device=self.device)
        outputs = self.model(
            input_ids, output_hidden_states=True, return_dict=True
        )
        h_surr = outputs.hidden_states[self.surrogate_layer + 1][0]  # [seq_len, d_surr]
        h_aligned = self.aligner.transform(h_surr)  # [seq_len, d_tgt]
        return h_aligned

    @torch.no_grad()
    def compute_prefix_cache_and_prior(
        self, prefix_ids: List[int], top_k: int
    ) -> PrefixCache:
        if len(prefix_ids) == 0:
            bos_id = self.tokenizer.bos_token_id or 1
            input_ids = torch.tensor([[bos_id]], device=self.device)
        else:
            input_ids = torch.tensor([prefix_ids], device=self.device)

        outputs = self.model(input_ids, use_cache=True, return_dict=True)
        past_kv = outputs.past_key_values

        logits = outputs.logits[0, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        top_k_values, top_k_indices = torch.topk(log_probs, min(top_k, self.vocab_size))

        prior_candidates = [
            (idx.item(), lp.item())
            for idx, lp in zip(top_k_indices, top_k_values)
        ]
        prior_log_probs = {c[0]: c[1] for c in prior_candidates}

        return PrefixCache(
            prefix_ids=prefix_ids.copy() if prefix_ids else [],
            past_key_values=past_kv,
            prior_candidates=prior_candidates,
            prior_log_probs=prior_log_probs,
        )

    def _expand_kv_cache(self, past_kv: Any, batch_size: int) -> DynamicCache:
        if hasattr(past_kv, 'to_legacy_cache'):
            legacy_cache = past_kv.to_legacy_cache()
        else:
            legacy_cache = past_kv

        expanded_cache = DynamicCache()
        for layer_idx in range(self.num_layers):
            key, value = legacy_cache[layer_idx]
            expanded_key = key.expand(batch_size, -1, -1, -1).contiguous()
            expanded_value = value.expand(batch_size, -1, -1, -1).contiguous()
            expanded_cache.update(expanded_key, expanded_value, layer_idx)

        return expanded_cache

    @torch.no_grad()
    def compute_hidden_distances_with_cache(
        self,
        prefix_cache: PrefixCache,
        candidate_ids: List[int],
        target_h: torch.Tensor,
        batch_size: int = 256,
        normalize: bool = True,
    ) -> List[float]:
        """
        Model + layercandidate token .

        Grey box: aligner.transform().
        """
        all_distances = []
        norm_factor = (torch.norm(target_h).item() + 1e-8) if normalize else 1.0

        for start_idx in range(0, len(candidate_ids), batch_size):
            end_idx = min(start_idx + batch_size, len(candidate_ids))
            batch_candidates = candidate_ids[start_idx:end_idx]
            current_batch_size = len(batch_candidates)

            candidate_input = torch.tensor(
                batch_candidates, device=self.device
            ).unsqueeze(1)

            if prefix_cache.past_key_values is not None:
                expanded_cache = self._expand_kv_cache(
                    prefix_cache.past_key_values, current_batch_size
                )
            else:
                expanded_cache = None

            outputs = self.model(
                candidate_input,
                past_key_values=expanded_cache,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )

            h_surr_batch = outputs.hidden_states[self.surrogate_layer + 1][:, -1, :]
            h_aligned_batch = self.aligner.transform(h_surr_batch)

            distances = torch.norm(h_aligned_batch - target_h.unsqueeze(0), dim=1)
            distances = distances / norm_factor
            all_distances.extend(distances.tolist())

        return all_distances

    @torch.no_grad()
    def get_embed_candidates(self, target_h_prev: torch.Tensor, top_k: int) -> List[int]:
        h = target_h_prev.float().cpu()
        lm_head = self.lm_head.float().cpu()
        scores = torch.matmul(lm_head, h)
        top_k_indices = torch.topk(scores, min(top_k, self.vocab_size)).indices
        return top_k_indices.tolist()


class BlackboxSearch:
    """
    Black box.

    Uses the same logic as the grey-box ColdstartSearch,
    The extractor is BlackboxExtractor instead of HiddenStateExtractor.
    """

    def __init__(
        self,
        extractor: BlackboxExtractor,
        K_prior: int = 50,
        K_embed: int = 0,
        beam_size: int = 1,
        lambda_prior: float = 0.0,
        coldstart_batch_size: int = 1000,
        candidate_batch_size: int = 256,
    ):
        self.extractor = extractor
        self.tokenizer = extractor.tokenizer
        self.vocab_size = extractor.vocab_size
        self.K_prior = K_prior
        self.K_embed = K_embed
        self.beam_size = beam_size
        self.lambda_prior = lambda_prior
        self.coldstart_batch_size = coldstart_batch_size
        self.candidate_batch_size = candidate_batch_size
        self.alpha = 1.0

    def _coldstart_first_token_topk(
        self, target_h: torch.Tensor, top_k: int
    ) -> List[Tuple[int, float]]:
        device = self.extractor.device
        norm_factor = torch.norm(target_h).item() + 1e-8
        all_distances = []

        for start_idx in range(0, self.vocab_size, self.coldstart_batch_size):
            end_idx = min(start_idx + self.coldstart_batch_size, self.vocab_size)

            with torch.no_grad():
                batch_ids = torch.arange(start_idx, end_idx, device=device).unsqueeze(1)
                outputs = self.extractor.model(
                    batch_ids, output_hidden_states=True, return_dict=True
                )
                h_surr = outputs.hidden_states[self.extractor.surrogate_layer + 1][:, 0, :]
                h_aligned = self.extractor.aligner.transform(h_surr)

                distances = torch.norm(h_aligned - target_h.unsqueeze(0), dim=1)
                distances = distances / norm_factor

                all_distances.extend([
                    (start_idx + i, distances[i].item())
                    for i in range(len(distances))
                ])

        sorted_candidates = sorted(all_distances, key=lambda x: x[1])
        return sorted_candidates[:top_k]

    def _compute_delta_score(self, distance: float, log_prob: float) -> float:
        return self.alpha * distance - self.lambda_prior * log_prob

    def search(
        self,
        target_hidden_states: torch.Tensor,
        token_ids: List[int],
        verbose: bool = True,
    ) -> SearchOutput:
        """
        Black box.

        Grey box difference: target_hidden_states are provided externally (intercepted),
        from Modelto.

        Args:
            target_hidden_states: Target hidden states [seq_len, hidden_size]
            token_ids: target token IDs(Used foraccuracy)
            verbose: whether to 
        """
        if len(token_ids) == 0:
            return SearchOutput([], [], "", 0.0, 0.0)

        target_length = len(token_ids)
        token_strs = [self.tokenizer.decode([t]) for t in token_ids]
        target_text = self.tokenizer.decode(token_ids)
        search_start = time.time()

        if verbose:
            logger.info(
                f"  [Black box] len={target_length}, B={self.beam_size}, "
                f"K_prior={self.K_prior}, K_embed={self.K_embed}"
            )

        # === position 0:cold start ===
        target_h_0 = target_hidden_states[0]
        coldstart_candidates = self._coldstart_first_token_topk(target_h_0, self.beam_size)

        beams: List[BeamState] = []
        for token_id, distance in coldstart_candidates:
            score = self._compute_delta_score(distance, 0.0)
            beams.append(BeamState(
                prefix_ids=[token_id], score=score,
                distances=[distance], log_probs=[0.0],
            ))

        if verbose:
            correct = any(b.prefix_ids[0] == token_ids[0] for b in beams)
            status = 'OK' if correct else 'MISS'
            top_tok = self.tokenizer.decode([beams[0].prefix_ids[0]])
            logger.info(f"    cold start: Top-1='{top_tok}' [{status}]")

        # === position 1..T-1 ===
        for pos in range(1, target_length):
            target_h = target_hidden_states[pos]
            target_h_prev = target_hidden_states[pos - 1]

            unique_prefixes: Dict[tuple, List[int]] = {}
            for beam in beams:
                pk = beam.get_prefix_key()
                if pk not in unique_prefixes:
                    unique_prefixes[pk] = beam.prefix_ids

            prefix_caches = {}
            for pk, pids in unique_prefixes.items():
                prefix_caches[pk] = self.extractor.compute_prefix_cache_and_prior(
                    pids, self.K_prior
                )

            all_candidates = []
            for beam_idx, beam in enumerate(beams):
                pk = beam.get_prefix_key()
                cache = prefix_caches[pk]

                prior_ids = set(c[0] for c in cache.prior_candidates)
                log_prob_dict = cache.prior_log_probs.copy()

                embed_ids: Set[int] = set()
                if self.K_embed > 0:
                    embed_ids = set(self.extractor.get_embed_candidates(target_h_prev, self.K_embed))

                candidate_ids = list(prior_ids | embed_ids)
                for cid in candidate_ids:
                    if cid not in log_prob_dict:
                        log_prob_dict[cid] = -100.0

                distances = self.extractor.compute_hidden_distances_with_cache(
                    cache, candidate_ids, target_h, self.candidate_batch_size
                )

                for i, cid in enumerate(candidate_ids):
                    d = distances[i]
                    lp = log_prob_dict[cid]
                    delta = self._compute_delta_score(d, lp)
                    all_candidates.append((beam_idx, cid, d, lp, beam.score + delta))

            all_candidates.sort(key=lambda x: x[4])
            top = all_candidates[:self.beam_size]

            new_beams = []
            for beam_idx, cid, d, lp, new_score in top:
                nb = beams[beam_idx].copy()
                nb.prefix_ids.append(cid)
                nb.score = new_score
                nb.distances.append(d)
                nb.log_probs.append(lp)
                new_beams.append(nb)
            beams = new_beams

            if verbose and (pos % 5 == 0 or pos == target_length - 1):
                accs = [
                    sum(1 for i, t in enumerate(b.prefix_ids) if i < len(token_ids) and t == token_ids[i])
                    / len(b.prefix_ids)
                    for b in beams
                ]
                logger.info(
                    f"    pos {pos+1}/{target_length} | Best={accs[0]*100:.1f}% Oracle={max(accs)*100:.1f}%"
                )

        # === Build results ===
        beam_results = []
        for rank, beam in enumerate(beams):
            avg_d = sum(beam.distances) / len(beam.distances)
            avg_lp = sum(beam.log_probs) / len(beam.log_probs)
            correct = sum(1 for i, t in enumerate(beam.prefix_ids)
                          if i < len(token_ids) and t == token_ids[i])
            acc = correct / len(token_ids)

            details = []
            for i, t in enumerate(beam.prefix_ids):
                if i < len(token_ids):
                    details.append({
                        'pos': i,
                        'target': token_strs[i],
                        'target_id': token_ids[i],
                        'recovered': self.tokenizer.decode([t]),
                        'recovered_id': t,
                        'distance': beam.distances[i],
                        'log_prob': beam.log_probs[i],
                        'correct': t == token_ids[i],
                    })

            beam_results.append(BeamResult(
                rank=rank, tokens=beam.prefix_ids,
                text=self.tokenizer.decode(beam.prefix_ids),
                score=beam.score, avg_distance=avg_d,
                avg_log_prob=avg_lp, token_accuracy=acc,
                token_details=details,
            ))

        best_acc = beam_results[0].token_accuracy if beam_results else 0.0
        oracle_acc = max(b.token_accuracy for b in beam_results) if beam_results else 0.0

        if verbose:
            elapsed = time.time() - search_start
            logger.info(f"    completed | {elapsed:.1f}s | Best={best_acc*100:.1f}% Oracle={oracle_acc*100:.1f}%")

        return SearchOutput(
            beam_results=beam_results,
            target_tokens=token_ids,
            target_text=target_text,
            best_accuracy=best_acc,
            oracle_accuracy=oracle_acc,
        )
