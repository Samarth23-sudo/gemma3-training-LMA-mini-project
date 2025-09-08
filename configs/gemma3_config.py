"""
Gemma 3 model configurations adapted for custom tokenizer
"""

# Note: This will import from the official Google Gemma repository
# You'll need to clone the repository and install it first
try:
    from gemma.config import GemmaConfig, Architecture, AttentionType
except ImportError:
    print("Warning: Official Gemma package not found. Please install it first:")
    print("git clone https://github.com/google/gemma_pytorch.git")
    print("cd gemma_pytorch && pip install -e .")
    
    # Fallback definitions (simplified versions)
    from dataclasses import dataclass
    from enum import Enum
    from typing import Optional, Sequence, Dict
    
    class Architecture(Enum):
        GEMMA_1 = "GEMMA_1"
        GEMMA_2 = "GEMMA_2" 
        GEMMA_3 = "GEMMA_3"
    
    class AttentionType(Enum):
        GLOBAL = "GLOBAL"
        LOCAL_SLIDING = "LOCAL_SLIDING"
    
    @dataclass
    class GemmaConfig:
        architecture: Architecture = Architecture.GEMMA_3
        vocab_size: int = 256000
        max_position_embeddings: int = 8192
        num_hidden_layers: int = 28
        num_attention_heads: int = 16
        num_key_value_heads: int = 16
        hidden_size: int = 3072
        intermediate_size: int = 24576
        head_dim: int = 256
        rms_norm_eps: float = 1e-6
        dtype: str = 'bfloat16'
        quant: bool = False
        tokenizer: Optional[str] = None
        attn_types: Optional[Sequence[AttentionType]] = None
        sliding_window_size: Optional[int] = None
        final_logit_softcapping: Optional[float] = None
        attn_logit_softcapping: Optional[float] = None
        query_pre_attn_scalar: Optional[int] = None
        use_pre_ffw_norm: bool = False
        use_post_ffw_norm: bool = False
        rope_wave_length: Optional[Dict[AttentionType, int]] = None
        use_qk_norm: bool = False
        vision_config = None
        rope_scaling_factor: Optional[int] = None


def get_custom_gemma3_200m_config(vocab_size: int, tokenizer_path: str) -> GemmaConfig:
    """
    Custom Gemma 3 configuration for ~200M parameters
    Optimized for multilingual training with custom tokenizer
    
    Architecture choices for 200M parameters:
    - 12 layers (reduced from 28)
    - 768 hidden size (reduced from 3072)  
    - 12 attention heads (reduced from 16)
    - 3072 intermediate size (reduced from 24576)
    """
    return GemmaConfig(
        # Model architecture
        architecture=Architecture.GEMMA_3,
        vocab_size=vocab_size,
        max_position_embeddings=4096,  # Reduced from 8192 for efficiency
        
        # Core architecture - optimized for ~200M params
        num_hidden_layers=12,           # Key reduction: 12 instead of 28 layers
        hidden_size=768,               # Reduced: 768 instead of 3072
        num_attention_heads=12,        # Reduced: 12 instead of 16 heads
        num_key_value_heads=12,        # Match attention heads
        head_dim=64,                   # 768 / 12 = 64
        intermediate_size=3072,        # 4x hidden_size (768 * 4)
        
        # Normalization
        rms_norm_eps=1e-6,
        
        # Data type and quantization
        dtype='bfloat16',
        quant=False,
        
        # Tokenizer
        tokenizer=tokenizer_path,
        
        # Attention configuration for Gemma 3
        attn_types=[AttentionType.GLOBAL] * 12,  # All global attention for smaller model
        
        # Gemma 3 specific features
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        query_pre_attn_scalar=144,  # sqrt(head_dim^2) for head_dim=64
        
        # Layer norm configuration
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        
        # RoPE configuration
        rope_wave_length={AttentionType.GLOBAL: 10000}
    )


def get_custom_gemma3_150m_config(vocab_size: int, tokenizer_path: str) -> GemmaConfig:
    """
    Custom Gemma 3 configuration for ~150M parameters
    Even more compact for faster training
    
    Architecture choices for 150M parameters:
    - 10 layers 
    - 640 hidden size
    - 10 attention heads
    - 2560 intermediate size
    """
    return GemmaConfig(
        # Model architecture
        architecture=Architecture.GEMMA_3,
        vocab_size=vocab_size,
        max_position_embeddings=2048,  # Further reduced for efficiency
        
        # Core architecture - optimized for ~150M params
        num_hidden_layers=10,           # Very compact: 10 layers
        hidden_size=640,               # Compact: 640 
        num_attention_heads=10,        # 10 heads
        num_key_value_heads=10,        # Match attention heads
        head_dim=64,                   # 640 / 10 = 64
        intermediate_size=2560,        # 4x hidden_size (640 * 4)
        
        # Normalization
        rms_norm_eps=1e-6,
        
        # Data type and quantization
        dtype='bfloat16',
        quant=False,
        
        # Tokenizer
        tokenizer=tokenizer_path,
        
        # Attention configuration for Gemma 3
        attn_types=[AttentionType.GLOBAL] * 10,  # All global attention
        
        # Gemma 3 specific features
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        query_pre_attn_scalar=144,  # sqrt(head_dim^2) for head_dim=64
        
        # Layer norm configuration
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        
        # RoPE configuration
        rope_wave_length={AttentionType.GLOBAL: 10000}
    )


def get_custom_gemma3_1b_config(vocab_size: int, tokenizer_path: str) -> GemmaConfig:
    """
    Custom 1B Gemma 3 configuration adapted for your tokenizer
    
    Args:
        vocab_size: Size of your custom vocabulary
        tokenizer_path: Path to your SentencePiece tokenizer
        
    Returns:
        GemmaConfig instance
    """
    return GemmaConfig(
        # Model architecture
        architecture=Architecture.GEMMA_3,
        dtype='bfloat16',
        
        # Model size (1B parameters)
        num_hidden_layers=26,
        num_attention_heads=4,
        num_key_value_heads=1,
        hidden_size=1152,
        intermediate_size=6912,
        head_dim=256,
        
        # Attention configuration
        attn_types=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING, 
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
        sliding_window_size=512,  # Smaller window for 1B model
        rope_wave_length={
            AttentionType.LOCAL_SLIDING: 10_000,
            AttentionType.GLOBAL: 1_000_000,
        },
        
        # Custom tokenizer settings
        vocab_size=vocab_size,
        tokenizer=tokenizer_path,
        
        # Context length
        max_position_embeddings=32_768,  # 32K context for Gemma 3
        
        # Normalization
        rms_norm_eps=1e-6,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        use_qk_norm=True,
        
        # Training settings
        quant=False,  # No quantization during training
        vision_config=None,  # Text-only model
        rope_scaling_factor=8,
        
        # Attention logit softcapping (important for Gemma 3)
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
    )


def get_custom_gemma3_2b_config(vocab_size: int, tokenizer_path: str) -> GemmaConfig:
    """
    Custom 2B Gemma 3 configuration (if you have enough GPU memory)
    
    Args:
        vocab_size: Size of your custom vocabulary
        tokenizer_path: Path to your SentencePiece tokenizer
        
    Returns:
        GemmaConfig instance
    """
    return GemmaConfig(
        # Model architecture
        architecture=Architecture.GEMMA_3,
        dtype='bfloat16',
        
        # Model size (2B parameters)
        num_hidden_layers=32,
        num_attention_heads=8,
        num_key_value_heads=4,
        hidden_size=1536,
        intermediate_size=9216,
        head_dim=256,
        
        # Attention configuration
        attn_types=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
        sliding_window_size=1024,  # Larger window for 2B model
        rope_wave_length={
            AttentionType.LOCAL_SLIDING: 10_000,
            AttentionType.GLOBAL: 1_000_000,
        },
        
        # Custom tokenizer settings
        vocab_size=vocab_size,
        tokenizer=tokenizer_path,
        
        # Context length
        max_position_embeddings=32_768,
        
        # Normalization
        rms_norm_eps=1e-6,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        use_qk_norm=True,
        
        # Training settings
        quant=False,
        vision_config=None,
        rope_scaling_factor=8,
        
        # Attention logit softcapping
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
    )


def get_training_config(model_size: str = "1b", kaggle_optimized: bool = True):
    """
    Get training configuration optimized for Kaggle
    
    Args:
        model_size: Model size ("150m", "200m", "1b", or "2b")
        kaggle_optimized: Whether to use Kaggle-optimized settings
        
    Returns:
        Training configuration object
    """
    
    class TrainingConfig:
        pass
    
    config = TrainingConfig()
    
    if model_size == "150m":
        # 150M model settings - fastest training, least memory
        config.learning_rate = 3e-4  # Higher LR for smaller model
        config.min_lr = 3e-6
        config.weight_decay = 0.01
        config.warmup_steps = 1000   # Less warmup needed
        config.max_steps = 30000 if not kaggle_optimized else 15000
        config.batch_size = 8 if not kaggle_optimized else 4  # Larger batch possible
        config.gradient_accumulation_steps = 4 if not kaggle_optimized else 8
        config.max_length = 2048
        
    elif model_size == "200m":
        # 200M model settings - good balance of speed and capacity
        config.learning_rate = 2.5e-4
        config.min_lr = 2.5e-6
        config.weight_decay = 0.01
        config.warmup_steps = 1500
        config.max_steps = 35000 if not kaggle_optimized else 18000
        config.batch_size = 6 if not kaggle_optimized else 3
        config.gradient_accumulation_steps = 6 if not kaggle_optimized else 12
        config.max_length = 2048
        
    elif model_size == "1b":
        # 1B model settings
        config.learning_rate = 2e-4
        config.min_lr = 2e-6
        config.weight_decay = 0.01
        config.warmup_steps = 2000
        config.max_steps = 50000 if not kaggle_optimized else 20000
        config.batch_size = 4 if not kaggle_optimized else 2
        config.gradient_accumulation_steps = 8 if not kaggle_optimized else 16
        config.max_length = 2048
        
    elif model_size == "2b":
        # 2B model settings (requires more memory)
        config.learning_rate = 1.5e-4
        config.min_lr = 1.5e-6
        config.weight_decay = 0.01
        config.warmup_steps = 3000
        config.max_steps = 75000 if not kaggle_optimized else 30000
        config.batch_size = 2 if not kaggle_optimized else 1
        config.gradient_accumulation_steps = 16 if not kaggle_optimized else 32
        config.max_length = 2048
    
    else:
        raise ValueError(f"Unsupported model size: {model_size}")
    
    # Common settings
    config.betas = (0.9, 0.95)
    config.eps = 1e-8
    config.use_amp = True  # Use automatic mixed precision
    config.max_grad_norm = 1.0
    config.save_every = 2000 if not kaggle_optimized else 1000
    config.eval_every = 1000 if not kaggle_optimized else 500
    config.log_every = 100
    
    # Kaggle-specific optimizations
    if kaggle_optimized:
        config.num_workers = 2  # Reduce CPU usage
        config.pin_memory = True
        config.compile_model = False  # May cause issues on Kaggle
        config.dataloader_prefetch_factor = 2
    
    return config


def validate_config(config: GemmaConfig, tokenizer_vocab_size: int) -> bool:
    """
    Validate that the configuration is consistent
    
    Args:
        config: GemmaConfig to validate
        tokenizer_vocab_size: Actual vocabulary size from tokenizer
        
    Returns:
        True if valid, raises ValueError if not
    """
    # Check vocab size match
    if config.vocab_size != tokenizer_vocab_size:
        raise ValueError(
            f"Config vocab_size ({config.vocab_size}) doesn't match "
            f"tokenizer vocab_size ({tokenizer_vocab_size})"
        )
    
    # Check attention heads divisibility
    if config.hidden_size % config.num_attention_heads != 0:
        raise ValueError(
            f"hidden_size ({config.hidden_size}) must be divisible by "
            f"num_attention_heads ({config.num_attention_heads})"
        )
    
    # Check head_dim consistency
    expected_head_dim = config.hidden_size // config.num_attention_heads
    if config.head_dim != expected_head_dim:
        raise ValueError(
            f"head_dim ({config.head_dim}) doesn't match "
            f"hidden_size / num_attention_heads ({expected_head_dim})"
        )
    
    # Check KV heads
    if config.num_key_value_heads > config.num_attention_heads:
        raise ValueError(
            f"num_key_value_heads ({config.num_key_value_heads}) cannot be greater than "
            f"num_attention_heads ({config.num_attention_heads})"
        )
    
    # Check attention patterns
    if config.attn_types and len(config.attn_types) == 0:
        raise ValueError("attn_types cannot be empty")
    
    print("✅ Configuration validation passed!")
    return True


def estimate_memory_usage(config: GemmaConfig, batch_size: int, sequence_length: int) -> dict:
    """
    Estimate memory usage for the given configuration
    
    Args:
        config: Model configuration
        batch_size: Training batch size
        sequence_length: Input sequence length
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Model parameters
    embedding_params = config.vocab_size * config.hidden_size
    
    layer_params = config.num_hidden_layers * (
        # Attention weights
        config.hidden_size * config.hidden_size * 3 +  # QKV projection
        config.hidden_size * config.hidden_size +      # Output projection
        # MLP weights  
        config.hidden_size * config.intermediate_size * 2 +  # Gate and up
        config.intermediate_size * config.hidden_size +      # Down
        # Layer norms
        config.hidden_size * 2
    )
    
    final_layer_params = config.hidden_size + config.vocab_size * config.hidden_size
    
    total_params = embedding_params + layer_params + final_layer_params
    
    # Memory estimates (assuming bfloat16 = 2 bytes per parameter)
    model_memory_gb = total_params * 2 / (1024**3)
    
    # Activation memory (rough estimate)
    activation_memory_gb = (
        batch_size * sequence_length * config.hidden_size * 
        config.num_hidden_layers * 4 * 2  # 4 activation tensors per layer, 2 bytes each
    ) / (1024**3)
    
    # Optimizer memory (AdamW stores 2 states per parameter)
    optimizer_memory_gb = total_params * 2 * 4 / (1024**3)  # 4 bytes for fp32 states
    
    # Total memory with overhead
    total_memory_gb = (model_memory_gb + activation_memory_gb + optimizer_memory_gb) * 1.2
    
    return {
        'model_params_millions': total_params / 1e6,
        'model_memory_gb': model_memory_gb,
        'activation_memory_gb': activation_memory_gb,
        'optimizer_memory_gb': optimizer_memory_gb,
        'total_memory_gb': total_memory_gb,
        'recommended_gpu_memory_gb': total_memory_gb + 2  # Extra headroom
    }


if __name__ == "__main__":
    # Example usage
    print("🔧 Testing Gemma 3 Configuration")
    print("=" * 50)
    
    # Test 1B config
    vocab_size = 32000  # Example vocab size
    tokenizer_path = "/path/to/tokenizer.model"
    
    config_1b = get_custom_gemma3_1b_config(vocab_size, tokenizer_path)
    print(f"1B Config: {config_1b.num_hidden_layers} layers, {config_1b.hidden_size} hidden size")
    
    # Memory estimation
    training_config = get_training_config("1b", kaggle_optimized=True)
    memory_est = estimate_memory_usage(config_1b, training_config.batch_size, training_config.max_length)
    
    print(f"Estimated memory usage:")
    print(f"  Model parameters: {memory_est['model_params_millions']:.1f}M")
    print(f"  Total GPU memory needed: {memory_est['recommended_gpu_memory_gb']:.1f}GB")
    print(f"  Kaggle GPU (16GB): {'✅ Should fit' if memory_est['recommended_gpu_memory_gb'] < 14 else '❌ May not fit'}")
    
    print("\n✅ Configuration module loaded successfully!")
