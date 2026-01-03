"""
Speculative Decoding Core Module

This module implements speculative decoding for accelerating LLM inference.

Available Generators:
- SpeculativeGenerator: Basic linear speculative decoding
- SpeculativeGeneratorWithStaticCache: Linear with pre-allocated cache
- StreamingSpeculativeGenerator: Linear with StreamingLLM compression
- TreeSpeculativeGenerator: Tree-based speculative decoding (SpecInfer-style)
- TreeSpeculativeGeneratorV2: Optimized tree-based with probability pruning
- TreeStreamingSpeculativeGenerator: Tree-based with StreamingLLM compression
- TreeStreamingSpeculativeGeneratorV2: Tree V2 + StreamingLLM (NEW)

Quantization Support:
- load_model_int8: Load model with INT8 quantization
- load_model_int4: Load model with INT4 quantization
- load_models_for_spec_decode: Convenience loader for model pairs
"""

from .static_cache import StaticKVCache
from .speculative_generator import SpeculativeGenerator, SpeculativeGeneratorWithStaticCache
from .streaming_speculative_generator import (
    StreamingSpeculativeGenerator,
    StreamingSpeculativeGeneratorV2
)
from .token_tree import TokenTree, TreeNode, build_tree_from_topk
from .tree_speculative_generator import (
    TreeSpeculativeGenerator,
    TreeSpeculativeGeneratorV2,
    TreeStreamingSpeculativeGenerator,
    TreeStreamingSpeculativeGeneratorV2
)
from .quantized_generator import (
    BITSANDBYTES_AVAILABLE,
    load_model_int8,
    load_model_int4,
    load_model_fp16,
    get_model_memory_footprint,
    compare_model_sizes,
    QuantizationConfig,
    load_models_for_spec_decode
)

__all__ = [
    # Cache utilities
    "StaticKVCache",
    
    # Linear speculative decoding
    "SpeculativeGenerator",
    "SpeculativeGeneratorWithStaticCache",
    "StreamingSpeculativeGenerator",
    "StreamingSpeculativeGeneratorV2",
    
    # Tree-based speculative decoding
    "TokenTree",
    "TreeNode",
    "build_tree_from_topk",
    "TreeSpeculativeGenerator",
    "TreeSpeculativeGeneratorV2",
    "TreeStreamingSpeculativeGenerator",
    "TreeStreamingSpeculativeGeneratorV2",
    
    # Quantization support
    "BITSANDBYTES_AVAILABLE",
    "load_model_int8",
    "load_model_int4",
    "load_model_fp16",
    "get_model_memory_footprint",
    "compare_model_sizes",
    "QuantizationConfig",
    "load_models_for_spec_decode",
]



