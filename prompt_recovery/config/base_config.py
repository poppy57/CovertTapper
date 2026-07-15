"""Configuration dataclasses for grey-box prompt inversion.

The module defines model, search, data, and experiment settings,
plus helpers for loading/saving YAML configuration files.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml
import os


@dataclass
class ModelConfig:
    """ModelConfig"""
    # SupportsModel
    SUPPORTED_MODELS = {
        "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
        "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",
        "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "vicuna-7b": "lmsys/vicuna-7b-v1.5",
        "vicuna-13b": "lmsys/vicuna-13b-v1.5",
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    }
    
    name: str = "llama2-7b"  # Model alias
    target_layer: int = -1   # Target layer (-1 means the final layer)
    torch_dtype: str = "float16"  # data type
    device_map: str = "auto"  # device map
    
    @property
    def model_path(self) -> str:
        """Return the HuggingFace model path or custom model path."""
        if self.name in self.SUPPORTED_MODELS:
            return self.SUPPORTED_MODELS[self.name]
        # Assume a custom path
        return self.name
    
    def get_target_layer(self, num_layers: int) -> int:
        """Resolve the effective target layer index."""
        if self.target_layer < 0:
            return num_layers + self.target_layer
        return min(self.target_layer, num_layers - 1)


@dataclass
class SearchConfig:
    """Search configuration"""
    # Candidate-pool configuration
    K_prior: int = 1000      # language-prior candidate pool size
    K_embed: int = 0         # embedding-similarity candidate pool size(0 meansdisabled)
    
    # parameter
    beam_size: int = 1       # beam size(1 = greedy search)
    lambda_prior: float = 0  # language-prior weight
    
    # Batching configuration
    coldstart_batch_size: int = 1000  # batch size for cold-start vocabulary traversal
    candidate_batch_size: int = 256   # batch size for candidate evaluation
    
    # Distance calculation
    distance_metric: str = "l2_normalized"  # l2, l2_normalized, cosine


@dataclass
class DataConfig:
    """Data configuration"""
    SUPPORTED_DATASETS = {
        "skytrax": "data_cache/skytrax_airline_63.csv",
        "cms": "data_cache/cms_medical.csv",
        "echr": "data_cache/echr_cases.csv",
        "private_pii": "data_cache/private_pii.csv",
        "wikitext": "data_cache/wikitext.csv",
        "custom": None,
    }
    
    dataset: str = "skytrax"  # Dataset
    data_path: Optional[str] = None  # custom data path
    max_samples: Optional[int] = None  # Maximum number of samples (None = all)
    max_length: Optional[int] = None   # maximum token length(None = unlimited)
    text_column: str = "content"  # text column name
    
    @property
    def dataset_path(self) -> str:
        """Dataset"""
        if self.data_path:
            return self.data_path
        if self.dataset in self.SUPPORTED_DATASETS:
            return self.SUPPORTED_DATASETS[self.dataset]
        raise ValueError(f"Unknown dataset: {self.dataset}")


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    # experiment name
    name: str = "prompt_inversion"
    
    # Path configuration
    output_dir: str = "results"
    log_dir: str = "logs"
    
    # Logging configuration
    log_level: str = "INFO"
    save_detailed_results: bool = True
    
    # random seed
    seed: Optional[int] = 42


@dataclass
class Config:
    """Top-level configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from a YAML file"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from a dictionary"""
        model_cfg = ModelConfig(**config_dict.get('model', {}))
        search_cfg = SearchConfig(**config_dict.get('search', {}))
        data_cfg = DataConfig(**config_dict.get('data', {}))
        exp_cfg = ExperimentConfig(**config_dict.get('experiment', {}))
        return cls(model=model_cfg, search=search_cfg, data=data_cfg, experiment=exp_cfg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary"""
        return {
            'model': {
                'name': self.model.name,
                'target_layer': self.model.target_layer,
                'torch_dtype': self.model.torch_dtype,
            },
            'search': {
                'K_prior': self.search.K_prior,
                'K_embed': self.search.K_embed,
                'beam_size': self.search.beam_size,
                'lambda_prior': self.search.lambda_prior,
                'coldstart_batch_size': self.search.coldstart_batch_size,
                'candidate_batch_size': self.search.candidate_batch_size,
                'distance_metric': self.search.distance_metric,
            },
            'data': {
                'dataset': self.data.dataset,
                'max_samples': self.data.max_samples,
                'max_length': self.data.max_length,
            },
            'experiment': {
                'name': self.experiment.name,
                'output_dir': self.experiment.output_dir,
            }
        }
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to a YAML file"""
        os.makedirs(os.path.dirname(yaml_path) or '.', exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
