"""
Quantized Model Support for Speculative Decoding

This module provides INT8 quantization support for tree-based speculative decoding,
enabling faster inference and reduced memory usage.

Key Features:
1. INT8 model loading via bitsandbytes
2. Compatible with all existing speculative decoding generators
3. Memory-efficient inference for large models

Requirements:
    pip install bitsandbytes

Usage:
    >>> from spec_decode.core.quantized_generator import load_model_int8
    >>> target_model = load_model_int8("/path/to/model")
    >>> generator = TreeSpeculativeGeneratorV2(target_model, draft_model, tokenizer)
"""

import torch
from typing import Optional, Dict, Any, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

# Check for bitsandbytes availability
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    BitsAndBytesConfig = None


def check_bitsandbytes_available():
    """Check if bitsandbytes is available for quantization."""
    if not BITSANDBYTES_AVAILABLE:
        raise ImportError(
            "bitsandbytes is required for INT8 quantization. "
            "Install it with: pip install bitsandbytes"
        )


def load_model_int8(
    model_path: str,
    device_map: str = "auto",
    llm_int8_threshold: float = 6.0,
    llm_int8_skip_modules: Optional[list] = None,
    llm_int8_enable_fp32_cpu_offload: bool = False,
    trust_remote_code: bool = False,
    **kwargs
) -> PreTrainedModel:
    """
    Load a model with INT8 quantization using bitsandbytes.
    
    INT8 quantization reduces memory usage by ~50% while maintaining
    similar inference quality. Throughput improvement depends on GPU.
    
    Args:
        model_path: Path to the model (local or HuggingFace Hub)
        device_map: Device mapping strategy ("auto", "cuda", etc.)
        llm_int8_threshold: Outlier threshold for mixed-precision
        llm_int8_skip_modules: Modules to skip during quantization
        llm_int8_enable_fp32_cpu_offload: Enable CPU offload for large models
        trust_remote_code: Trust remote code for custom models
        **kwargs: Additional arguments passed to from_pretrained
        
    Returns:
        Quantized model ready for inference
        
    Example:
        >>> model = load_model_int8("/mnt/disk1/models/pythia-2.8b")
        >>> # Use with speculative decoding
        >>> gen = TreeSpeculativeGeneratorV2(model, draft_model, tokenizer)
    """
    check_bitsandbytes_available()
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=llm_int8_threshold,
        llm_int8_skip_modules=llm_int8_skip_modules,
        llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **kwargs
    )
    
    return model


def load_model_int4(
    model_path: str,
    device_map: str = "auto",
    bnb_4bit_compute_dtype: torch.dtype = torch.float16,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
    trust_remote_code: bool = False,
    **kwargs
) -> PreTrainedModel:
    """
    Load a model with INT4 quantization using bitsandbytes (QLoRA-style).
    
    INT4 quantization reduces memory usage by ~75% but may have 
    slightly lower quality than INT8.
    
    Args:
        model_path: Path to the model (local or HuggingFace Hub)
        device_map: Device mapping strategy
        bnb_4bit_compute_dtype: Compute dtype for 4-bit base models
        bnb_4bit_quant_type: Quantization type ("fp4" or "nf4")
        bnb_4bit_use_double_quant: Use nested quantization
        trust_remote_code: Trust remote code for custom models
        **kwargs: Additional arguments passed to from_pretrained
        
    Returns:
        Quantized model ready for inference
        
    Example:
        >>> model = load_model_int4("/mnt/disk1/models/pythia-2.8b")
    """
    check_bitsandbytes_available()
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **kwargs
    )
    
    return model


def load_model_fp16(
    model_path: str,
    device_map: str = "auto",
    trust_remote_code: bool = False,
    **kwargs
) -> PreTrainedModel:
    """
    Load a model in FP16 precision (baseline, no quantization).
    
    Args:
        model_path: Path to the model
        device_map: Device mapping strategy
        trust_remote_code: Trust remote code
        **kwargs: Additional arguments
        
    Returns:
        Model in FP16 precision
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **kwargs
    )
    
    return model


def get_model_memory_footprint(model: PreTrainedModel) -> Dict[str, float]:
    """
    Get memory footprint of a model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with memory statistics in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    # Get GPU memory if available
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**2
    
    return {
        "param_size_mb": param_size / 1024**2,
        "buffer_size_mb": buffer_size / 1024**2,
        "total_size_mb": (param_size + buffer_size) / 1024**2,
        "gpu_allocated_mb": gpu_memory,
    }


def compare_model_sizes(fp16_model: PreTrainedModel, quant_model: PreTrainedModel) -> Dict[str, Any]:
    """
    Compare memory footprint between FP16 and quantized models.
    
    Args:
        fp16_model: Model in FP16 precision
        quant_model: Quantized model (INT8 or INT4)
        
    Returns:
        Comparison statistics
    """
    fp16_stats = get_model_memory_footprint(fp16_model)
    quant_stats = get_model_memory_footprint(quant_model)
    
    compression_ratio = fp16_stats["total_size_mb"] / max(quant_stats["total_size_mb"], 1e-6)
    memory_saved = fp16_stats["total_size_mb"] - quant_stats["total_size_mb"]
    
    return {
        "fp16": fp16_stats,
        "quantized": quant_stats,
        "compression_ratio": compression_ratio,
        "memory_saved_mb": memory_saved,
        "memory_saved_percent": (memory_saved / fp16_stats["total_size_mb"]) * 100 if fp16_stats["total_size_mb"] > 0 else 0,
    }


class QuantizationConfig:
    """Configuration class for quantization settings."""
    
    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    
    @classmethod
    def get_loader(cls, quant_type: str):
        """Get the appropriate model loader for quantization type."""
        loaders = {
            cls.INT8: load_model_int8,
            cls.INT4: load_model_int4,
            cls.FP16: load_model_fp16,
        }
        if quant_type not in loaders:
            raise ValueError(f"Unknown quantization type: {quant_type}. "
                           f"Supported: {list(loaders.keys())}")
        return loaders[quant_type]


def load_models_for_spec_decode(
    target_model_path: str,
    draft_model_path: str,
    target_quant: str = "fp16",
    draft_quant: str = "fp16",
    device_map: str = "auto",
    **kwargs
) -> tuple:
    """
    Load both target and draft models with specified quantization.
    
    This is a convenience function for loading model pairs for
    speculative decoding with different quantization configurations.
    
    Args:
        target_model_path: Path to target (large) model
        draft_model_path: Path to draft (small) model
        target_quant: Quantization for target ("fp16", "int8", "int4")
        draft_quant: Quantization for draft ("fp16", "int8", "int4")
        device_map: Device mapping strategy
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (target_model, draft_model, tokenizer)
        
    Example:
        >>> target, draft, tokenizer = load_models_for_spec_decode(
        ...     "/mnt/disk1/models/pythia-2.8b",
        ...     "/mnt/disk1/models/pythia-70m",
        ...     target_quant="int8",
        ...     draft_quant="fp16"
        ... )
    """
    # Load tokenizer from target model
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load target model with specified quantization
    target_loader = QuantizationConfig.get_loader(target_quant)
    target_model = target_loader(target_model_path, device_map=device_map, **kwargs)
    
    # Load draft model with specified quantization
    draft_loader = QuantizationConfig.get_loader(draft_quant)
    draft_model = draft_loader(draft_model_path, device_map=device_map, **kwargs)
    
    return target_model, draft_model, tokenizer


__all__ = [
    "BITSANDBYTES_AVAILABLE",
    "check_bitsandbytes_available",
    "load_model_int8",
    "load_model_int4",
    "load_model_fp16",
    "get_model_memory_footprint",
    "compare_model_sizes",
    "QuantizationConfig",
    "load_models_for_spec_decode",
]

