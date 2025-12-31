"""
Speculative Decoding Generator

This module implements the main speculative decoding algorithm for accelerating
LLM inference using a small draft model to propose tokens and a large target
model to verify them.

Algorithm:
1. Draft model generates K tokens quickly (using temporary cache within round)
2. Target model verifies all K tokens in parallel (one forward pass)
3. Accept tokens where draft matches target (greedy), reject at first mismatch
4. Update target cache and continue

Key Optimizations:
- Direct use of HuggingFace's DynamicCache (no conversion overhead)
- Temporary cache for draft model (discarded between rounds)
- torch.compile for reduced Python overhead
- torch.inference_mode for disabled autograd
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.cache_utils import DynamicCache


class SpeculativeGenerator:
    """
    Speculative Decoding Generator for accelerated LLM inference.
    
    This generator uses a small draft model to propose tokens and a larger
    target model to verify them. The draft model's cache is temporary (per-round),
    while the target model uses a persistent DynamicCache.
    
    Args:
        target_model: Large model for verification (e.g., Pythia-2.8B)
        draft_model: Small model for drafting (e.g., Pythia-70M)
        tokenizer: Tokenizer for the models
        K: Number of tokens to draft per round (default: 5)
        max_len: Maximum total sequence length (default: 2048)
        device: Device to run on (default: "cuda")
        use_compile: Whether to use torch.compile optimization (default: True)
    """
    
    def __init__(
        self,
        target_model: PreTrainedModel,
        draft_model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        K: int = 5,
        max_len: int = 2048,
        device: str = "cuda",
        use_compile: bool = True
    ):
        self.device = device
        self.K = K
        self.max_len = max_len
        
        # Store models
        self.target_model = target_model.to(device)
        self.draft_model = draft_model.to(device)
        self.tokenizer = tokenizer
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set models to eval mode
        self.target_model.eval()
        self.draft_model.eval()
        
        # Apply torch.compile if available and requested
        self._compile_enabled = False
        if use_compile and hasattr(torch, 'compile'):
            try:
                # Configure compile options for dynamic cache lengths
                # Note: "reduce-overhead" mode uses CUDA Graphs which can conflict
                # with dynamic cache states in speculative decoding. Use "default"
                # mode for better compatibility with KV cache operations.
                compile_kwargs = {
                    "mode": "default",           # Safer mode, avoids CUDA Graph issues
                    "fullgraph": False,          # Allow partial graph compilation
                    "dynamic": True,             # Key: Support dynamic cache lengths
                }
                
                self.target_model = torch.compile(
                    self.target_model, 
                    **compile_kwargs
                )
                self.draft_model = torch.compile(
                    self.draft_model,
                    **compile_kwargs
                )
                self._compile_enabled = True
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}")
                self._compile_enabled = False
        
        # Cache for target model - use HuggingFace's DynamicCache directly
        self.target_cache: Optional[DynamicCache] = None
        
        # Generation state
        self.current_ids: Optional[torch.Tensor] = None
        self._last_target_logits: Optional[torch.Tensor] = None
        
        # Statistics tracking
        self.stats = {
            "total_tokens": 0,
            "total_accepted": 0,
            "total_rounds": 0,
            "total_draft_tokens": 0,
        }
    
    def reset(self):
        """Reset the generator state for a new generation session."""
        self.target_cache = None
        self.current_ids = None
        self._last_target_logits = None
        self.stats = {
            "total_tokens": 0,
            "total_accepted": 0,
            "total_rounds": 0,
            "total_draft_tokens": 0,
        }
    
    @torch.inference_mode()
    def _prefill(self, input_ids: torch.Tensor):
        """
        Prefill phase: process the initial prompt with target model.
        
        Args:
            input_ids: Initial prompt tokens [1, prompt_len]
        """
        outputs = self.target_model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True
        )
        
        # Store the cache directly (HuggingFace's DynamicCache)
        self.target_cache = outputs.past_key_values
        self.current_ids = input_ids
        
        # Store logits for predicting the first new token
        self._last_target_logits = outputs.logits[:, -1:, :]  # [1, 1, vocab]
    
    @torch.inference_mode()
    def _draft_k_tokens(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draft K tokens using the small draft model.
        
        The draft model uses a temporary cache that is discarded after drafting.
        This avoids complex cache synchronization between models.
        
        Returns:
            draft_tokens: Drafted tokens [1, K]
            draft_logits: Logits for each drafted token [K, vocab_size]
        """
        draft_tokens = []
        draft_logits_list = []
        
        # Re-prefill draft model with current sequence
        # This is efficient because draft model is small
        draft_outputs = self.draft_model(
            input_ids=self.current_ids,
            use_cache=True,
            return_dict=True
        )
        temp_draft_cache = draft_outputs.past_key_values
        next_token_logits = draft_outputs.logits[:, -1, :]  # [1, vocab]
        
        # Draft K tokens autoregressively
        for _ in range(self.K):
            # Greedy selection
            next_token = next_token_logits.argmax(dim=-1)  # [1]
            draft_tokens.append(next_token)
            draft_logits_list.append(next_token_logits.squeeze(0))  # [vocab]
            
            # Forward the new token to get next prediction
            draft_outputs = self.draft_model(
                input_ids=next_token.unsqueeze(0),  # [1, 1]
                past_key_values=temp_draft_cache,
                use_cache=True,
                return_dict=True
            )
            temp_draft_cache = draft_outputs.past_key_values
            next_token_logits = draft_outputs.logits[:, -1, :]  # [1, vocab]
        
        # Stack results
        draft_tokens = torch.stack(draft_tokens, dim=1)  # [1, K]
        draft_logits = torch.stack(draft_logits_list, dim=0)  # [K, vocab]
        
        return draft_tokens, draft_logits
    
    @torch.inference_mode()
    def _verify_tokens(self, draft_tokens: torch.Tensor) -> Tuple[torch.Tensor, object]:
        """
        Verify K draft tokens with the target model in one forward pass.
        
        Args:
            draft_tokens: Tokens to verify [1, K]
            
        Returns:
            target_logits: Logits for verification [K+1, vocab_size]
                          - [0] verifies draft[0] (from _last_target_logits)
                          - [1] verifies draft[1]
                          - ...
                          - [K] is for bonus token if all accepted
            verify_outputs: Original model outputs (used for cache optimization)
        """
        # Forward all K draft tokens with target's cache
        outputs = self.target_model(
            input_ids=draft_tokens,
            past_key_values=self.target_cache,
            use_cache=True,
            return_dict=True
        )
        
        # Update cache (will be rolled back if needed)
        self.target_cache = outputs.past_key_values
        
        # Combine with last logits:
        # _last_target_logits predicts draft[0]
        # outputs.logits[0, i, :] predicts draft[i+1] (or bonus if i == K-1)
        new_logits = outputs.logits[0]  # [K, vocab]
        target_logits = torch.cat([
            self._last_target_logits.squeeze(0),  # [1, vocab] -> [1, vocab]
            new_logits  # [K, vocab]
        ], dim=0)  # [K+1, vocab]
        
        return target_logits, outputs
    
    def _accept_reject_greedy(
        self,
        draft_tokens: torch.Tensor,
        target_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, int, bool]:
        """
        Greedy acceptance strategy.
        
        Compare draft tokens with target predictions. Accept tokens where they match,
        reject at first mismatch and use target's prediction instead.
        
        Args:
            draft_tokens: Draft tokens [1, K]
            target_logits: Target logits [K+1, vocab_size]
            
        Returns:
            accepted_tokens: Tokens to add [1, num_accepted]
            num_accepted: Number of accepted tokens (1 to K+1)
            all_accepted: Whether all K draft tokens were accepted
        """
        K = draft_tokens.shape[1]
        accepted_tokens = []
        all_accepted = True  # Assume all accepted until we find a mismatch
        
        for i in range(K):
            target_pred = target_logits[i].argmax(dim=-1)  # scalar
            draft_tok = draft_tokens[0, i]  # scalar
            
            if target_pred.item() == draft_tok.item():
                # Draft matches target - accept the draft token
                accepted_tokens.append(draft_tok)
            else:
                # Mismatch - accept target's prediction and stop
                accepted_tokens.append(target_pred)
                all_accepted = False  # We had a mismatch
                break
        
        num_accepted = len(accepted_tokens)
        
        # If all K draft tokens accepted, add bonus token
        if all_accepted:
            bonus_token = target_logits[K].argmax(dim=-1)
            accepted_tokens.append(bonus_token)
            num_accepted += 1
        
        # Stack into tensor
        accepted_tensor = torch.stack(accepted_tokens, dim=0).unsqueeze(0)  # [1, num_accepted]
        
        return accepted_tensor, num_accepted, all_accepted
    
    @torch.inference_mode()
    def _update_cache_and_logits(
        self,
        num_accepted: int,
        accepted_tokens: torch.Tensor,
        all_accepted: bool,
        original_len: int,
        verify_outputs: Optional[object] = None
    ):
        """
        Update target cache and _last_target_logits after accept/reject.
        
        Optimized batch update strategy:
        - all_accepted: Cache already has K tokens from verify, only need to
          forward the bonus token to get its KV into cache
        - partial reject: Rollback cache and batch forward all accepted tokens
          in a single pass (already optimized)
        
        Args:
            num_accepted: Number of tokens accepted
            accepted_tokens: The accepted tokens [1, num_accepted]
            all_accepted: Whether all K draft tokens were accepted
            original_len: Cache length before this round
            verify_outputs: Original outputs from verify phase (for potential reuse)
        """
        if all_accepted:
            # All K draft tokens accepted + bonus
            # Cache already has K tokens from verify, need to add bonus token's KV
            # Note: We still need to forward the bonus token to add its KV to cache
            bonus_token = accepted_tokens[:, -1:]  # [1, 1]
            
            outputs = self.target_model(
                input_ids=bonus_token,
                past_key_values=self.target_cache,
                use_cache=True,
                return_dict=True
            )
            
            self.target_cache = outputs.past_key_values
            self._last_target_logits = outputs.logits  # [1, 1, vocab]
        else:
            # Some draft tokens were rejected
            # Optimized: Batch forward ALL accepted tokens in a single pass
            # This is more efficient than forwarding tokens one by one
            
            # Rollback cache to before this round
            self.target_cache.crop(original_len)
            
            # Batch forward ALL accepted tokens to ensure correct cache
            # HuggingFace models automatically handle causal masking correctly
            outputs = self.target_model(
                input_ids=accepted_tokens,  # [1, num_accepted] - batch update
                past_key_values=self.target_cache,
                use_cache=True,
                return_dict=True
            )
            
            self.target_cache = outputs.past_key_values
            self._last_target_logits = outputs.logits[:, -1:, :]  # [1, 1, vocab]
    
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        verbose: bool = False
    ) -> str:
        """
        Generate text using speculative decoding.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            verbose: Whether to print progress information
            
        Returns:
            Generated text (including prompt)
        """
        # Reset state
        self.reset()
        
        # Tokenize prompt
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len - max_new_tokens
        ).input_ids.to(self.device)
        
        # Prefill
        self._prefill(input_ids)
        
        if verbose:
            print(f"Prefilled {input_ids.shape[1]} tokens")
        
        generated = 0
        eos_token_id = self.tokenizer.eos_token_id
        
        while generated < max_new_tokens:
            # Check if approaching max length
            if self.current_ids.shape[1] + self.K >= self.max_len:
                break
            
            # Record cache length before this round
            original_cache_len = self.target_cache.get_seq_length()
            
            # Phase 1: Draft K tokens
            draft_tokens, draft_logits = self._draft_k_tokens()
            self.stats["total_draft_tokens"] += self.K
            
            # Phase 2: Verify with target model
            target_logits, verify_outputs = self._verify_tokens(draft_tokens)
            
            # Phase 3: Accept/Reject
            accepted_tokens, num_accepted, all_accepted = self._accept_reject_greedy(
                draft_tokens, target_logits
            )
            
            # Limit to max_new_tokens
            remaining = max_new_tokens - generated
            if num_accepted > remaining:
                accepted_tokens = accepted_tokens[:, :remaining]
                num_accepted = remaining
                all_accepted = False
            
            if num_accepted == 0:
                break
            
            # Phase 4: Update cache and logits (optimized batch update)
            self._update_cache_and_logits(
                num_accepted, accepted_tokens, all_accepted, original_cache_len,
                verify_outputs=verify_outputs
            )
            
            # Update sequence
            self.current_ids = torch.cat([self.current_ids, accepted_tokens], dim=-1)
            self.stats["total_accepted"] += num_accepted
            self.stats["total_tokens"] += num_accepted
            self.stats["total_rounds"] += 1
            generated += num_accepted
            
            if verbose:
                print(f"Round {self.stats['total_rounds']}: accepted {num_accepted}/{self.K+1}, total: {generated}")
            
            # Check for EOS
            if accepted_tokens[0, -1].item() == eos_token_id:
                if verbose:
                    print("Generated EOS token, stopping.")
                break
        
        return self.tokenizer.decode(self.current_ids[0], skip_special_tokens=True)
    
    def get_stats(self) -> dict:
        """Get generation statistics."""
        stats = dict(self.stats)
        if stats["total_draft_tokens"] > 0:
            stats["acceptance_rate"] = stats["total_accepted"] / stats["total_draft_tokens"]
        else:
            stats["acceptance_rate"] = 0.0
        
        if stats["total_rounds"] > 0:
            stats["tokens_per_round"] = stats["total_tokens"] / stats["total_rounds"]
        else:
            stats["tokens_per_round"] = 0.0
        
        return stats
    
    def get_acceptance_rate(self) -> float:
        """Get the acceptance rate of draft tokens."""
        if self.stats["total_draft_tokens"] > 0:
            return self.stats["total_accepted"] / self.stats["total_draft_tokens"]
        return 0.0


# Try to import StaticCache from transformers (available in newer versions)
try:
    from transformers import StaticCache
    HAS_STATIC_CACHE = True
except ImportError:
    HAS_STATIC_CACHE = False


class SpeculativeGeneratorWithStaticCache(SpeculativeGenerator):
    """
    Speculative Decoding Generator using HuggingFace's StaticCache.
    
    This variant uses pre-allocated StaticCache which:
    - Avoids dynamic memory allocation
    - Enables O(1) truncation operations
    - Compatible with torch.compile max-autotune mode
    
    Note: Requires transformers >= 4.38.0 for StaticCache support.
    
    Args:
        target_model: Large model for verification
        draft_model: Small model for drafting
        tokenizer: Tokenizer for the models
        K: Number of tokens to draft per round
        max_cache_len: Maximum cache length (pre-allocated)
        device: Device to run on
        use_max_autotune: Whether to use torch.compile with max-autotune mode
    """
    
    def __init__(
        self,
        target_model: PreTrainedModel,
        draft_model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        K: int = 5,
        max_cache_len: int = 2048,
        device: str = "cuda",
        use_max_autotune: bool = True
    ):
        if not HAS_STATIC_CACHE:
            raise ImportError(
                "StaticCache requires transformers >= 4.38.0. "
                "Please upgrade: pip install --upgrade transformers"
            )
        
        # Initialize parent class without torch.compile
        super().__init__(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            K=K,
            max_len=max_cache_len,
            device=device,
            use_compile=False  # We handle compilation ourselves
        )
        
        self.max_cache_len = max_cache_len
        
        # Create StaticCache for target model
        self.target_static_cache = StaticCache(
            config=self.target_model.config,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device=device,
            dtype=torch.float16
        )
        
        # Apply torch.compile with max-autotune if requested
        # StaticCache enables this because shapes are fixed
        self._compile_enabled = False
        if use_max_autotune and hasattr(torch, 'compile'):
            try:
                self.target_model = torch.compile(
                    self.target_model,
                    mode="max-autotune",  # More aggressive optimization
                    fullgraph=False
                )
                self.draft_model = torch.compile(
                    self.draft_model,
                    mode="max-autotune",
                    fullgraph=False
                )
                self._compile_enabled = True
                print("✅ torch.compile with max-autotune enabled (StaticCache mode)")
            except Exception as e:
                print(f"Warning: torch.compile with max-autotune failed: {e}")
                self._compile_enabled = False
    
    def reset(self):
        """Reset the generator state for a new generation session."""
        # Reset StaticCache instead of DynamicCache
        self.target_static_cache.reset()
        self.target_cache = None  # Will be set during prefill
        self.current_ids = None
        self._last_target_logits = None
        self.stats = {
            "total_tokens": 0,
            "total_accepted": 0,
            "total_rounds": 0,
            "total_draft_tokens": 0,
        }
    
    @torch.inference_mode()
    def _prefill(self, input_ids: torch.Tensor):
        """
        Prefill phase using StaticCache.
        
        Args:
            input_ids: Initial prompt tokens [1, prompt_len]
        """
        # Reset static cache before prefill
        self.target_static_cache.reset()
        
        outputs = self.target_model(
            input_ids=input_ids,
            past_key_values=self.target_static_cache,
            use_cache=True,
            return_dict=True
        )
        
        # StaticCache is updated in-place, also store reference
        self.target_cache = outputs.past_key_values
        self.current_ids = input_ids
        
        # Store logits for predicting the first new token
        self._last_target_logits = outputs.logits[:, -1:, :]
    
    @torch.inference_mode()
    def _verify_tokens(self, draft_tokens: torch.Tensor) -> Tuple[torch.Tensor, object]:
        """
        Verify K draft tokens using StaticCache.
        """
        outputs = self.target_model(
            input_ids=draft_tokens,
            past_key_values=self.target_cache,
            use_cache=True,
            return_dict=True
        )
        
        # Cache is updated in-place
        self.target_cache = outputs.past_key_values
        
        new_logits = outputs.logits[0]
        target_logits = torch.cat([
            self._last_target_logits.squeeze(0),
            new_logits
        ], dim=0)
        
        return target_logits, outputs
    
    @torch.inference_mode()
    def _update_cache_and_logits(
        self,
        num_accepted: int,
        accepted_tokens: torch.Tensor,
        all_accepted: bool,
        original_len: int,
        verify_outputs: Optional[object] = None
    ):
        """
        Update StaticCache after accept/reject.
        
        StaticCache supports efficient truncation via reset + recompute
        or by adjusting the seen_tokens counter (if available).
        """
        if all_accepted:
            # All accepted - just add bonus token's KV
            bonus_token = accepted_tokens[:, -1:]
            
            outputs = self.target_model(
                input_ids=bonus_token,
                past_key_values=self.target_cache,
                use_cache=True,
                return_dict=True
            )
            
            self.target_cache = outputs.past_key_values
            self._last_target_logits = outputs.logits
        else:
            # Partial reject - need to truncate cache
            # StaticCache doesn't have crop(), so we need to reset and recompute
            # This is a limitation compared to DynamicCache, but still efficient
            # because StaticCache uses fixed memory
            
            # For StaticCache, we need to reset and recompute from scratch
            # This is less efficient but maintains correctness
            # Note: We recompute only the valid prefix + accepted tokens
            
            # Get the valid sequence up to original_len + accepted tokens
            valid_seq = self.current_ids
            
            # Reset cache
            self.target_static_cache.reset()
            
            # Recompute cache for valid sequence + accepted tokens
            new_seq = torch.cat([valid_seq, accepted_tokens], dim=-1)
            
            outputs = self.target_model(
                input_ids=new_seq,
                past_key_values=self.target_static_cache,
                use_cache=True,
                return_dict=True
            )
            
            self.target_cache = outputs.past_key_values
            self._last_target_logits = outputs.logits[:, -1:, :]
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if StaticCache is available in the current transformers version."""
        return HAS_STATIC_CACHE
