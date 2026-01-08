import os
from typing import Dict, List, Any, Optional
class EvoMIConfig:
    """
    EvoMI Configuration Management Class
    Provides unified configuration management, including task configuration generation and model settings
    """
    
    def __init__(self):
        # Default configuration
        self._base_model = ['models/Qwen3-4B-Instruct-2507','models/Qwen3-4B-thinking-2507','models/Qwen3-4B-thinking-2507']
        self._expert_model = ['models/Qwen3-4B-thinking-2507', 'models/Qwen3-4B-Instruct-2507', 'models/Qwen3-4B-Instruct-2507']
        self._base_model_bi = ['models/Qwen3-4B-Instruct-2507','models/Qwen3-4B-thinking-2507']
        self._expert_model_bi = ['models/Qwen3-4B-thinking-2507', 'models/Qwen3-4B-Instruct-2507']
        self._checkpoint_dir = './checkpoints'
        self._default_max_tokens = 20000
        self._default_max_model_len = None
        
        # Ensure cache directory exists
        os.makedirs(self._checkpoint_dir, exist_ok=True)
    
    @property
    def base_model(self) -> List[str]:
        """Get base model path list"""
        return self._base_model
    
    @base_model.setter
    def base_model(self, value: List[str]) -> None:
        """Set base model path list"""
        self._base_model = value
    
    @property
    def expert_model(self) -> List[str]:
        """Get expert model path list"""
        return self._expert_model
    
    @expert_model.setter
    def expert_model(self, value: List[str]) -> None:
        """Set expert model path list"""
        self._expert_model = value
    
    @property
    def checkpoint_dir(self) -> str:
        """Get checkpoint directory"""
        return self._checkpoint_dir
    
    @checkpoint_dir.setter
    def checkpoint_dir(self, value: str) -> None:
        """Set checkpoint directory"""
        self._checkpoint_dir = value
        # Ensure directory exists
        os.makedirs(value, exist_ok=True)
    
    def create_biojective_eval_task_config(self, 
                                model_path: str,
                                max_tokens: int = None) -> Dict[str, Any]:
        """
        Create evaluation task configuration (for model_reproduction use)
        
        Args:
            model_path: Model path
            max_tokens: Maximum number of generated tokens
        
        Returns:
            TaskConfig object
        """
        from evalscope import TaskConfig
        
        # Use default value if not provided
        if max_tokens is None:
            max_tokens = self._default_max_tokens
        
        # Set different generation parameters based on model type
        if 'thinking' in model_path or 'merged' in model_path:
            # Thinking model parameters: T=0.6, Top-p=0.95
            generation_config={
                'max_tokens': max_tokens,
                'temperature': 0.6,
                'top_p': 0.95,
                'top_k': 20
            }
        else:
            # Instruct model parameters: T=0.7, Top-p=0.8, topk=50
            generation_config={
                'max_tokens': max_tokens,
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 20
            }
        
        # Create task configuration (excluding ifeval dataset)
        task_cfg = TaskConfig(
            model=model_path,
            api_url='',  # Will be updated according to port later
            eval_type='server',
            datasets=['aime25', 'gpqa_diamond'],
            dataset_args={
                'aime25': {
                    'aggregation': 'mean_and_pass_at_k',
                    'metric_list':[{'acc': {'numeric': True}},{'tokens_num': {'tokenizer_path': model_path}},'think_num']
                },
                'gpqa_diamond': {
                    'aggregation': 'mean_and_vote_at_k',
                    'metric_list':['acc',{'tokens_num': {'tokenizer_path': model_path}},'think_num']
                }
            },
            eval_batch_size=64,
            ignore_errors=True,
            generation_config=generation_config,
            timeout=60000,
            stream=True,
            #  aime25 question nums: 15+15, gpqa_diamond: 198, ifeval:541    
            repeats={'aime25':1,'gpqa_diamond':1},
            limit={'aime25':None,'gpqa_diamond':None}
        )
        return task_cfg

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model
        
        Parameters:
            model_name: Model name
        
        Returns:
            Model configuration dictionary
        """
        # Can return specific configurations for different models here
        return {
            "name": model_name,
            "checkpoint_dir": os.path.join(self._checkpoint_dir, model_name),
            "base_model": model_name in self._base_model
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format
        
        Returns:
            Configuration dictionary
        """
        return {
            "base_model": self._base_model,
            "expert_model": self._expert_model,
            "checkpoint_dir": self._checkpoint_dir,
            "default_max_tokens": self._default_max_tokens,
            "default_max_model_len": self._default_max_model_len
        }
    
    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Load configuration from dictionary
        
        Parameters:
            config_dict: Dictionary containing configuration
        """
        if "base_model" in config_dict:
            self._base_model = config_dict["base_model"]
        if "expert_model" in config_dict:
            self._expert_model = config_dict["expert_model"]
        if "checkpoint_dir" in config_dict:
            self._checkpoint_dir = config_dict["checkpoint_dir"]
        if "default_max_tokens" in config_dict:
            self._default_max_tokens = config_dict["default_max_tokens"]
        if "default_max_model_len" in config_dict:
            self._default_max_model_len = config_dict["default_max_model_len"]

# Create global configuration instance
config_manager = EvoMIConfig()