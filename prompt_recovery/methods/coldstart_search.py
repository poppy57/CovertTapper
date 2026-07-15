"""
Cold-start beam-search method

Core algorithm:
1. First position:cold start - scan the full vocabulary,select the nearest tokens by hidden-state distance top-B token
2. Later positions:mixed candidate pool -  Top-K +  Top-K
3. Supports beam search, B 

Scoring function:
    L(X) = sum_t alpha_t * d_t(X_1:t) - lambda * sum_t log Q(x_t | X_1:t-1)
    
where d_t is the hidden-state L2 distance,log Q the language-prior log probability

Optimizations:
- share KV-cache computation across beams
- reuse cache for language-prior scoring
"""

import torch
import logging
import time
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from copy import deepcopy

from models.hidden_state_extractor import HiddenStateExtractor, PrefixCache
from config import SearchConfig

logger = logging.getLogger(__name__)


@dataclass
class BeamState:
    """
    Beam state
    
    Stores a beam hypothesis as (X_1:t, state_F, state_Q, score)
    """
    prefix_ids: List[int]       # current prefix token IDs
    score: float                # cumulative cost L(X_1:t)
    distances: List[float]      # distance at each position d_t
    log_probs: List[float]      # position log Q(x_t | X_1:t-1)
    
    def copy(self) -> "BeamState":
        """Create a copy"""
        return BeamState(
            prefix_ids=self.prefix_ids.copy(),
            score=self.score,
            distances=self.distances.copy(),
            log_probs=self.log_probs.copy(),
        )
    
    def get_prefix_key(self) -> tuple:
        """Get a prefix hash key for deduplication"""
        return tuple(self.prefix_ids)


@dataclass
class BeamCandidate:
    """Candidate expansion"""
    beam_idx: int           # source beam index
    token_id: int           # candidate token ID
    token_str: str          # token string
    distance: float         # d_t
    log_prob: float         # log Q(x_t | X_1:t-1)
    delta_score: float      # incremental cost delta
    new_score: float        # cumulative cost


@dataclass
class BeamResult:
    """Final result for one beam"""
    rank: int                           # rank (0-based)
    tokens: List[int]                   #  token IDs
    text: str                           # recovered text
    score: float                        # cumulative cost (lower is better)
    avg_distance: float                 # average hidden-state distance
    avg_log_prob: float                 # language-prior log probability
    token_accuracy: float               #  token accuracy
    token_details: List[Dict] = field(default_factory=list)  # per-position details


@dataclass
class SearchOutput:
    """Search output"""
    beam_results: List[BeamResult]      # all beam results(sorted by score)
    target_tokens: List[int]            # target token IDs
    target_text: str                    # target text
    best_accuracy: float                #  beam accuracy
    oracle_accuracy: float              # any beam accuracy(upper bound)


class ColdstartSearch:
    """Cold-start beam search"""
    
    def __init__(
        self,
        extractor: HiddenStateExtractor,
        config: SearchConfig,
    ):
        """
        Initialize the searcher
        
        Args:
            extractor: Hidden-state extractor
            config: Search configuration
        """
        self.extractor = extractor
        self.config = config
        self.tokenizer = extractor.tokenizer
        self.vocab_size = extractor.vocab_size
        
        # alpha_t weight(simplified to 1.0)
        self.alpha = 1.0
    
    def _coldstart_first_token_topk(
        self,
        target_h: torch.Tensor,
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """
        cold start:scan the full vocabulary Top-K token
        
        Args:
            target_h: Target hidden states
            top_k: number of candidates to return
            
        Returns:
            [(token_id, distance), ...] sorted by distance
        """
        batch_size = self.config.coldstart_batch_size
        device = self.extractor.device
        norm_factor = torch.norm(target_h).item() + 1e-8
        
        all_distances = []
        
        for start_idx in range(0, self.vocab_size, batch_size):
            end_idx = min(start_idx + batch_size, self.vocab_size)
            
            with torch.no_grad():
                batch_ids = torch.arange(start_idx, end_idx, device=device).unsqueeze(1)
                
                outputs = self.extractor.model(
                    batch_ids,
                    output_hidden_states=True,
                    return_dict=True
                )
                hidden_states = outputs.hidden_states[self.extractor.target_layer + 1]
                h_batch = hidden_states[:, 0, :]
                
                distances = torch.norm(h_batch - target_h.unsqueeze(0), dim=1)
                distances = distances / norm_factor
                
                all_distances.extend([
                    (start_idx + i, distances[i].item())
                    for i in range(len(distances))
                ])
        
        # Sort and return Top-K
        sorted_candidates = sorted(all_distances, key=lambda x: x[1])
        return sorted_candidates[:top_k]
    
    def _compute_beam_caches(
        self,
        beams: List[BeamState],
    ) -> Dict[tuple, PrefixCache]:
        """
        Compute prefix caches for all beams(after deduplication)
        
        This optimization shares computation across beams
        
        Args:
            beams: current beam states
            
        Returns:
            {prefix_key: PrefixCache} dictionary
        """
        # Deduplicate:compute identical prefixes once
        unique_prefixes: Dict[tuple, List[int]] = {}
        for beam in beams:
            prefix_key = beam.get_prefix_key()
            if prefix_key not in unique_prefixes:
                unique_prefixes[prefix_key] = beam.prefix_ids
        
        # compute the cache for each unique prefix
        prefix_caches = {}
        for prefix_key, prefix_ids in unique_prefixes.items():
            cache = self.extractor.compute_prefix_cache_and_prior(
                prefix_ids, self.config.K_prior
            )
            prefix_caches[prefix_key] = cache
        
        return prefix_caches
    
    def _get_hybrid_candidates_from_cache(
        self,
        prefix_cache: PrefixCache,
        target_h_prev: torch.Tensor,
    ) -> Tuple[List[int], Dict[int, float], Dict[str, int]]:
        """
        from mixed candidate pool
        
        Args:
            prefix_cache: precomputed prefix cache
            target_h_prev: target hidden state at the previous position
            
        Returns:
            (all_candidate_ids, log_prob_dict, stats)
        """
        # 1. (already in the cache)
        prior_ids = set(c[0] for c in prefix_cache.prior_candidates)
        prior_log_probs = prefix_cache.prior_log_probs.copy()
        
        # 2. embedding-similarity candidates
        embed_ids: Set[int] = set()
        if self.config.K_embed > 0:
            embed_candidates = self.extractor.get_embed_candidates(
                target_h_prev, self.config.K_embed
            )
            embed_ids = set(embed_candidates)
        
        # 3. merge candidate pools
        all_ids = list(prior_ids | embed_ids)
        
        # embedding-similarity candidates log_prob
        for cid in all_ids:
            if cid not in prior_log_probs:
                prior_log_probs[cid] = -100.0
        
        # Statistics
        stats = {
            'prior_only': len(prior_ids - embed_ids),
            'embed_only': len(embed_ids - prior_ids),
            'overlap': len(prior_ids & embed_ids),
            'total': len(all_ids),
        }
        
        return all_ids, prior_log_probs, stats
    
    def _compute_delta_score(
        self,
        distance: float,
        log_prob: float,
    ) -> float:
        """
        incremental cost
        
        delta L_t = alpha_t * d_t - lambda * log Q(x_t | X_1:t-1)
        
        Args:
            distance: d_t
            log_prob: log Q
            
        Returns:
            incremental cost
        """
        return self.alpha * distance - self.config.lambda_prior * log_prob
    
    def _compute_beam_accuracy(self, beam: BeamState, target_ids: List[int]) -> float:
        """Compute current beam accuracy"""
        correct = sum(1 for i, t in enumerate(beam.prefix_ids) if i < len(target_ids) and t == target_ids[i])
        return correct / len(beam.prefix_ids) if beam.prefix_ids else 0.0
    
    def _format_progress_bar(self, current: int, total: int, width: int = 20) -> str:
        """Build a progress bar"""
        filled = int(width * current / total)
        bar = '#' * filled + '-' * (width - filled)
        return f"[{bar}] {current}/{total}"
    
    def search(
        self,
        token_ids: List[int],
        verbose: bool = True
    ) -> SearchOutput:
        """
        Run beam search
        
        Args:
            token_ids: target token ID sequence
            verbose: whether to print detailed logs
            
        Returns:
            SearchOutput all beam results
        """
        if len(token_ids) == 0:
            return SearchOutput(
                beam_results=[],
                target_tokens=[],
                target_text="",
                best_accuracy=0.0,
                oracle_accuracy=0.0,
            )
        
        beam_size = self.config.beam_size
        target_length = len(token_ids)
        token_strs = [self.tokenizer.decode([t]) for t in token_ids]
        target_text = self.tokenizer.decode(token_ids)
        batch_size = self.config.candidate_batch_size
        
        search_start_time = time.time()
        
        if verbose:
            logger.info(f"sequence length={target_length}, Beam={beam_size}, K_prior={self.config.K_prior}, K_embed={self.config.K_embed}")
        
        # Extract target hidden states
        target_hidden_states = self.extractor.extract_hidden_states(token_ids)
        
        # === position 0:cold start ===
        target_h_0 = target_hidden_states[0]
        target_token_id_0 = token_ids[0]
        
        coldstart_candidates = self._coldstart_first_token_topk(target_h_0, beam_size)
        
        # Initialize beams
        beams: List[BeamState] = []
        coldstart_correct = False
        for i, (token_id, distance) in enumerate(coldstart_candidates):
            log_prob = 0.0
            score = self._compute_delta_score(distance, log_prob)
            beams.append(BeamState(
                prefix_ids=[token_id],
                score=score,
                distances=[distance],
                log_probs=[log_prob],
            ))
            if token_id == target_token_id_0:
                coldstart_correct = True
        
        if verbose:
            status = 'OK' if coldstart_correct else 'MISS'
            top_token = self.tokenizer.decode([beams[0].prefix_ids[0]])
            logger.info(f"  cold start: Top-1='{top_token}' {status} | target in Top-{beam_size}: {'yes' if coldstart_correct else 'no'}")
        
        # === position 1 to T-1:beam search ===
        for pos in range(1, target_length):
            target_token_id = token_ids[pos]
            target_h = target_hidden_states[pos]
            target_h_prev = target_hidden_states[pos - 1]
            
            # Batch-compute prefix caches for all beams
            prefix_caches = self._compute_beam_caches(beams)
            
            # Candidate expansion
            all_candidates: List[BeamCandidate] = []
            
            for beam_idx, beam in enumerate(beams):
                prefix_key = beam.get_prefix_key()
                prefix_cache = prefix_caches[prefix_key]
                
                candidate_ids, log_prob_dict, stats = self._get_hybrid_candidates_from_cache(
                    prefix_cache, target_h_prev
                )
                
                distances = self.extractor.compute_hidden_distances_with_cache(
                    prefix_cache, candidate_ids, target_h, batch_size
                )
                
                for i, cid in enumerate(candidate_ids):
                    d = distances[i]
                    log_prob = log_prob_dict[cid]
                    delta = self._compute_delta_score(d, log_prob)
                    new_score = beam.score + delta
                    
                    all_candidates.append(BeamCandidate(
                        beam_idx=beam_idx,
                        token_id=cid,
                        token_str=self.tokenizer.decode([cid]),
                        distance=d,
                        log_prob=log_prob,
                        delta_score=delta,
                        new_score=new_score,
                    ))
            
            #  new_score ,select Top-B
            all_candidates.sort(key=lambda x: x.new_score)
            top_candidates = all_candidates[:beam_size]
            
            # Update beams
            new_beams: List[BeamState] = []
            for cand in top_candidates:
                old_beam = beams[cand.beam_idx]
                new_beam = old_beam.copy()
                new_beam.prefix_ids.append(cand.token_id)
                new_beam.score = cand.new_score
                new_beam.distances.append(cand.distance)
                new_beam.log_probs.append(cand.log_prob)
                new_beams.append(new_beam)
            
            beams = new_beams
            
            # Periodically print progress(5positionposition)
            if verbose and (pos % 5 == 0 or pos == target_length - 1):
                progress = self._format_progress_bar(pos + 1, target_length)
                
                #  beam 
                beam_accs = [self._compute_beam_accuracy(b, token_ids) for b in beams]
                best_acc = beam_accs[0]
                oracle_acc = max(beam_accs)
                
                # Show top-beam information
                top_beam = beams[0]
                top_preview = self.tokenizer.decode(top_beam.prefix_ids[-3:]) if len(top_beam.prefix_ids) >= 3 else self.tokenizer.decode(top_beam.prefix_ids)
                top_preview = top_preview.replace('\n', '<nl>')[:15]
                
                logger.info(f"  {progress} | Best={best_acc*100:.1f}% Oracle={oracle_acc*100:.1f}% | ...{top_preview}")
        
        # === Build final results ===
        beam_results: List[BeamResult] = []
        
        for rank, beam in enumerate(beams):
            # Statistics
            avg_distance = sum(beam.distances) / len(beam.distances) if beam.distances else 0.0
            avg_log_prob = sum(beam.log_probs) / len(beam.log_probs) if beam.log_probs else 0.0
            
            #  token accuracy
            correct_count = sum(
                1 for i, t in enumerate(beam.prefix_ids)
                if i < len(token_ids) and t == token_ids[i]
            )
            token_accuracy = correct_count / len(token_ids) if token_ids else 0.0
            
            # per-position details
            token_details = []
            for i, t in enumerate(beam.prefix_ids):
                if i < len(token_ids):
                    token_details.append({
                        'pos': i,
                        'target': token_strs[i] if i < len(token_strs) else '',
                        'target_id': token_ids[i] if i < len(token_ids) else -1,
                        'recovered': self.tokenizer.decode([t]),
                        'recovered_id': t,
                        'distance': beam.distances[i] if i < len(beam.distances) else 0.0,
                        'log_prob': beam.log_probs[i] if i < len(beam.log_probs) else 0.0,
                        'correct': t == token_ids[i] if i < len(token_ids) else False,
                    })
            
            beam_results.append(BeamResult(
                rank=rank,
                tokens=beam.prefix_ids,
                text=self.tokenizer.decode(beam.prefix_ids),
                score=beam.score,
                avg_distance=avg_distance,
                avg_log_prob=avg_log_prob,
                token_accuracy=token_accuracy,
                token_details=token_details,
            ))
        
        #  oracle accuracy
        best_accuracy = beam_results[0].token_accuracy if beam_results else 0.0
        oracle_accuracy = max(b.token_accuracy for b in beam_results) if beam_results else 0.0
        
        total_time = time.time() - search_start_time
        
        # Print a concise summary
        if verbose:
            logger.info(f"  completed! elapsed={total_time:.1f}s | Best={best_accuracy*100:.1f}% | Oracle={oracle_accuracy*100:.1f}%")
            
            #  beam 
            if beam_size > 1:
                beam_info = " | ".join([f"B{i}:{br.token_accuracy*100:.0f}%" for i, br in enumerate(beam_results)])
                logger.info(f"  Beamaccuracy: {beam_info}")
        
        return SearchOutput(
            beam_results=beam_results,
            target_tokens=token_ids,
            target_text=target_text,
            best_accuracy=best_accuracy,
            oracle_accuracy=oracle_accuracy,
        )
    
    def search_simple(
        self,
        token_ids: List[int],
        verbose: bool = True
    ) -> Dict:
        """
        Simplified interface:Returndictionary
        
        Args:
            token_ids:  token IDs
            verbose: 
            
        Returns:
            dictionary
        """
        output = self.search(token_ids, verbose)
        
        if not output.beam_results:
            return {
                'results': [],
                'accuracy': 0.0,
                'recovered_tokens': [],
                'recovered_text': '',
                'beam_results': [],
            }
        
        best = output.beam_results[0]
        
        return {
            'results': best.token_details,
            'accuracy': best.token_accuracy,
            'recovered_tokens': best.tokens,
            'recovered_text': best.text,
            'beam_results': [
                {
                    'rank': br.rank,
                    'tokens': br.tokens,
                    'text': br.text,
                    'score': br.score,
                    'accuracy': br.token_accuracy,
                    'avg_distance': br.avg_distance,
                }
                for br in output.beam_results
            ],
            'best_accuracy': output.best_accuracy,
            'oracle_accuracy': output.oracle_accuracy,
        }
