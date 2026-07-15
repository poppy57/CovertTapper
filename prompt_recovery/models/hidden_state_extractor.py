"""
Hidden-state extractor

Used for:
- extract target-sequence hidden states
- candidate token 
- batching for speed
- KV Cache reuse
"""

import torch
import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

logger = logging.getLogger(__name__)


@dataclass
class PrefixCache:
    """Prefix KV cache and related metadata"""
    prefix_ids: List[int]
    past_key_values: Any  # KV cache
    prior_candidates: List[Tuple[int, float]]  # [(token_id, log_prob), ...]
    prior_log_probs: Dict[int, float]  # {token_id: log_prob}


class HiddenStateExtractor:
    """Hidden-state extractor"""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        target_layer: int = -1,
    ):
        """
        Hidden-state extractor
        
        Args:
            model: TrainingModel
            tokenizer: tokenizer
            target_layer: layer(-1 meanslayer)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # layer
        num_layers = model.config.num_hidden_layers
        self.target_layer = target_layer if target_layer >= 0 else num_layers + target_layer
        self.num_layers = num_layers
        
        #  LM Head weightUsed for
        self.lm_head = model.lm_head.weight.detach()
        self.vocab_size = len(tokenizer)
        
        logger.info(f"Hidden-state extractor:")
        logger.info(f"  - layer: {self.target_layer}/{num_layers}")
        logger.info(f"  - LM Head dimension: {self.lm_head.shape}")
    
    @torch.no_grad()
    def extract_hidden_states(self, token_ids: List[int]) -> torch.Tensor:
        """
        Extract sequence hidden states
        
        Args:
            token_ids: token ID 
            
        Returns:
             [seq_len, hidden_size]
        """
        input_ids = torch.tensor([token_ids], device=self.device)
        outputs = self.model(
            input_ids,
            output_hidden_states=True,
            return_dict=True
        )
        # hidden_states[0] is the embedding layer,[1] is layer...
        hidden_states = outputs.hidden_states[self.target_layer + 1]
        return hidden_states[0]  # [seq_len, hidden_size]
    
    @torch.no_grad()
    def compute_prefix_cache_and_prior(
        self,
        prefix_ids: List[int],
        top_k: int
    ) -> PrefixCache:
        """
         KV cache  Top-K
        
        This is the key optimization:
        
        Args:
            prefix_ids:  token IDs
            top_k: 
            
        Returns:
            PrefixCache  cache 
        """
        if len(prefix_ids) == 0:
            # , BOS token
            bos_id = self.tokenizer.bos_token_id or 1
            input_ids = torch.tensor([[bos_id]], device=self.device)
        else:
            input_ids = torch.tensor([prefix_ids], device=self.device)
        
        # 
        outputs = self.model(
            input_ids,
            use_cache=True,
            return_dict=True
        )
        
        # extract KV cache
        past_kv = outputs.past_key_values
        
        # extract
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
    
    @torch.no_grad()
    def compute_hidden_distance(
        self,
        prefix_ids: List[int],
        candidate_id: int,
        target_h: torch.Tensor,
        normalize: bool = True
    ) -> float:
        """
        candidate token 
        
        Args:
            prefix_ids:  token IDs
            candidate_id: candidate token ID
            target_h: Target hidden states
            normalize: whether to 
            
        Returns:
            
        """
        new_ids = prefix_ids + [candidate_id]
        input_ids = torch.tensor([new_ids], device=self.device)
        outputs = self.model(
            input_ids,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.hidden_states[self.target_layer + 1]
        h = hidden_states[0, -1, :]
        
        distance = torch.norm(h - target_h).item()
        if normalize:
            norm_factor = torch.norm(target_h).item() + 1e-8
            distance = distance / norm_factor
        
        return distance
    
    def _expand_kv_cache(self, past_kv: Any, batch_size: int) -> DynamicCache:
        """
         KV cache to batch size
        
        Args:
            past_kv:  KV cache
            batch_size:  batch size
            
        Returns:
             DynamicCache
        """
        #  cache
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
        normalize: bool = True
    ) -> List[float]:
        """
         KV cache 
        
        Args:
            prefix_cache: precomputed prefix cache
            candidate_ids: candidate token ID 
            target_h: Target hidden states
            batch_size: 
            normalize: whether to 
            
        Returns:
            
        """
        all_distances = []
        norm_factor = (torch.norm(target_h).item() + 1e-8) if normalize else 1.0
        
        for start_idx in range(0, len(candidate_ids), batch_size):
            end_idx = min(start_idx + batch_size, len(candidate_ids))
            batch_candidates = candidate_ids[start_idx:end_idx]
            current_batch_size = len(batch_candidates)
            
            # candidate token 
            candidate_input = torch.tensor(
                batch_candidates, device=self.device
            ).unsqueeze(1)
            
            #  KV cache
            if prefix_cache.past_key_values is not None:
                expanded_cache = self._expand_kv_cache(
                    prefix_cache.past_key_values, current_batch_size
                )
            else:
                expanded_cache = None
            
            # 
            outputs = self.model(
                candidate_input,
                past_key_values=expanded_cache,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True
            )
            
            # extract
            hidden_states = outputs.hidden_states[self.target_layer + 1]
            h_batch = hidden_states[:, -1, :]
            
            # 
            distances = torch.norm(h_batch - target_h.unsqueeze(0), dim=1)
            distances = distances / norm_factor
            all_distances.extend(distances.tolist())
        
        return all_distances
    
    @torch.no_grad()
    def compute_hidden_distances_batch(
        self,
        prefix_ids: List[int],
        candidate_ids: List[int],
        target_h: torch.Tensor,
        batch_size: int = 256,
        normalize: bool = True
    ) -> List[float]:
        """
        candidate token ()
        
        Args:
            prefix_ids:  token IDs
            candidate_ids: candidate token ID 
            target_h: Target hidden states
            batch_size: 
            normalize: whether to 
            
        Returns:
            
        """
        #  cache
        prefix_cache = self.compute_prefix_cache_and_prior(prefix_ids, top_k=1)
        return self.compute_hidden_distances_with_cache(
            prefix_cache, candidate_ids, target_h, batch_size, normalize
        )
    
    @torch.no_grad()
    def get_prior_candidates(
        self,
        prefix_ids: List[int],
        top_k: int
    ) -> List[Tuple[int, float]]:
        """
         Top-K ()
        
        Args:
            prefix_ids:  token IDs
            top_k: 
            
        Returns:
            (token_id, log_prob) 
        """
        cache = self.compute_prefix_cache_and_prior(prefix_ids, top_k)
        return cache.prior_candidates
    
    @torch.no_grad()
    def get_embed_candidates(
        self,
        target_h_prev: torch.Tensor,
        top_k: int
    ) -> List[int]:
        """
         Top-K 
        
        Args:
            target_h_prev: target hidden state at the previous position
            top_k: 
            
        Returns:
            candidate token ID 
        """
        # to CPU 
        h = target_h_prev.float().cpu()
        lm_head = self.lm_head.float().cpu()
        
        # 
        scores = torch.matmul(lm_head, h)
        
        #  Top-K
        top_k_indices = torch.topk(scores, min(top_k, self.vocab_size)).indices
        return top_k_indices.tolist()
    
    @torch.no_grad()
    def get_token_prior_rank(
        self,
        prefix_ids: List[int],
        token_id: int
    ) -> int:
        """
         token rank
        
        Args:
            prefix_ids:  token IDs
            token_id:  token ID
            
        Returns:
            rank(from  1 )
        """
        if len(prefix_ids) == 0:
            bos_id = self.tokenizer.bos_token_id or 1
            input_ids = torch.tensor([[bos_id]], device=self.device)
        else:
            input_ids = torch.tensor([prefix_ids], device=self.device)
        
        outputs = self.model(input_ids)
        logits = outputs.logits[0, -1, :]
        sorted_indices = torch.argsort(logits, descending=True)
        rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1
        return rank
