"""
ModelLoad

Supports:
-  LLM ModelLoad
- automatic device placement
- HuggingFace authentication
"""

import os
import torch
import logging
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def setup_hf_auth(token: Optional[str] = None):
    """Set up HuggingFace authentication"""
    if token is None:
        token = os.environ.get("HF_TOKEN")
    
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        logger.info("HuggingFace authentication")
    else:
        logger.warning(" HuggingFace token,ModelLoad")


class ModelLoader:
    """ModelLoad"""
    
    DTYPE_MAP = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    
    def __init__(
        self,
        model_path: str,
        torch_dtype: str = "float16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ):
        """
        ModelLoad
        
        Args:
            model_path: Model HuggingFace Model
            torch_dtype: data type
            device_map: device map
            trust_remote_code: whether to 
        """
        self.model_path = model_path
        self.torch_dtype = self.DTYPE_MAP.get(torch_dtype, torch.float16)
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        
        self.model = None
        self.tokenizer = None
    
    def load(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer"""
        logger.info(f"Load model: {self.model_path}")
        
        # Loadtokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code
        )
        
        # Set the pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code
        )
        self.model.eval()
        
        # Model
        num_layers = self.model.config.num_hidden_layers
        hidden_size = self.model.config.hidden_size
        vocab_size = len(self.tokenizer)
        
        logger.info(f"ModelLoadcompleted:")
        logger.info(f"  - layer: {num_layers}")
        logger.info(f"  - dimension: {hidden_size}")
        logger.info(f"  - Vocabulary size: {vocab_size}")
        logger.info(f"  - Device: {next(self.model.parameters()).device}")
        
        return self.model, self.tokenizer
    
    @property
    def device(self) -> torch.device:
        """ModelDevice"""
        if self.model is None:
            raise RuntimeError("ModelLoad")
        return next(self.model.parameters()).device
    
    @property
    def num_layers(self) -> int:
        """model layer"""
        if self.model is None:
            raise RuntimeError("ModelLoad")
        return self.model.config.num_hidden_layers
    
    @property
    def hidden_size(self) -> int:
        """dimension"""
        if self.model is None:
            raise RuntimeError("ModelLoad")
        return self.model.config.hidden_size
    
    @property
    def vocab_size(self) -> int:
        """Vocabulary size"""
        if self.tokenizer is None:
            raise RuntimeError("tokenizerLoad")
        return len(self.tokenizer)
