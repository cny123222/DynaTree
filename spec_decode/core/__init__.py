"""
Speculative Decoding Core Module

This module implements speculative decoding for accelerating LLM inference.
"""

from .static_cache import StaticKVCache
from .speculative_generator import SpeculativeGenerator, SpeculativeGeneratorWithStaticCache

__all__ = ["StaticKVCache", "SpeculativeGenerator", "SpeculativeGeneratorWithStaticCache"]



