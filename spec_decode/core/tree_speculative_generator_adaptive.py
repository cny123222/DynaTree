"""
Adaptive Tree Speculative Generator.

This module provides adaptive tree structure implementations for speculative decoding:
- Phase 1: TreeSpeculativeGeneratorV2Adaptive - Confidence-based adaptive branching
- Phase 2: TreeSpeculativeGeneratorV2AdaptiveV2 - Dynamic depth with early stopping
- Phase 3: TreeSpeculativeGeneratorV2AdaptiveV3 - Historical acceptance rate adjustment

Key innovations:
1. Dynamic branch factor based on draft model confidence
2. Early stopping for low-probability branches
3. Runtime parameter adjustment based on historical acceptance rates
"""

import math
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .tree_speculative_generator import TreeSpeculativeGeneratorV2
from .token_tree import TokenTree


# =============================================================================
# Phase 1: Confidence-based Adaptive Branching
# =============================================================================

class TreeSpeculativeGeneratorV2Adaptive(TreeSpeculativeGeneratorV2):
    """
    Adaptive Tree Speculative Decoding with Confidence-based Branching.
    
    This generator dynamically adjusts the branch factor based on the draft
    model's confidence (top-1 probability):
    - High confidence (>0.8): Use fewer branches (model is certain)
    - Medium confidence (0.3-0.8): Use normal branches
    - Low confidence (<0.3): Use more branches (explore more options)
    
    Args:
        high_conf_threshold: Threshold for high confidence (default: 0.8)
        low_conf_threshold: Threshold for low confidence (default: 0.3)
        min_branch: Minimum branch factor for high confidence (default: 1)
        max_branch: Maximum branch factor for low confidence (default: 4)
        default_branch: Default branch factor for medium confidence (default: 2)
    """
    
    def __init__(
        self,
        target_model,
        draft_model,
        tokenizer,
        tree_depth: int = 4,
        branch_factor: int = 2,  # Used as default/medium confidence branch
        max_tree_nodes: int = 128,
        max_len: int = 2048,
        device: str = "cuda",
        use_compile: bool = True,
        probability_threshold: float = 0.05,
        # Adaptive parameters
        high_conf_threshold: float = 0.8,
        low_conf_threshold: float = 0.3,
        min_branch: int = 1,
        max_branch: int = 4,
    ):
        super().__init__(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            tree_depth=tree_depth,
            branch_factor=branch_factor,
            max_tree_nodes=max_tree_nodes,
            max_len=max_len,
            device=device,
            use_compile=use_compile,
            probability_threshold=probability_threshold
        )
        
        # Adaptive branching parameters
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold
        self.min_branch = min_branch
        self.max_branch = max_branch
        self.default_branch = branch_factor
        
        # Statistics for adaptive branching
        self.adaptive_stats = {
            "high_conf_count": 0,
            "medium_conf_count": 0,
            "low_conf_count": 0,
            "total_expansions": 0,
        }
    
    def reset(self):
        """Reset generator state including adaptive stats."""
        super().reset()
        self.adaptive_stats = {
            "high_conf_count": 0,
            "medium_conf_count": 0,
            "low_conf_count": 0,
            "total_expansions": 0,
        }
    
    def _get_adaptive_branch_factor(self, logits: torch.Tensor) -> int:
        """
        Determine branch factor based on draft model confidence.
        
        Args:
            logits: Logits from draft model [vocab_size]
            
        Returns:
            Adaptive branch factor
        """
        # Calculate confidence as max probability
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max().item()
        
        self.adaptive_stats["total_expansions"] += 1
        
        if confidence > self.high_conf_threshold:
            self.adaptive_stats["high_conf_count"] += 1
            return self.min_branch
        elif confidence < self.low_conf_threshold:
            self.adaptive_stats["low_conf_count"] += 1
            return self.max_branch
        else:
            self.adaptive_stats["medium_conf_count"] += 1
            return self.default_branch
    
    @torch.inference_mode()
    def _draft_tree_tokens(self) -> TokenTree:
        """
        Generate tree with adaptive branching based on confidence.
        
        At each node expansion, the branch factor is determined by the
        draft model's confidence in its prediction.
        """
        tree = TokenTree(
            max_depth=self.tree_depth,
            branch_factor=self.max_branch,  # Use max for capacity
            max_nodes=self.max_tree_nodes,
            device=self.device
        )
        
        # Re-prefill draft model
        draft_outputs = self.draft_model(
            input_ids=self.current_ids,
            use_cache=True,
            return_dict=True
        )
        
        first_logits = draft_outputs.logits[:, -1, :]
        
        # Get adaptive branch factor for root
        root_branch = self._get_adaptive_branch_factor(first_logits[0])
        
        log_probs = F.log_softmax(first_logits, dim=-1)
        topk_probs, topk_tokens = torch.topk(log_probs[0], root_branch)
        
        # Add root (top-1 token)
        tree.add_root(topk_tokens[0].item(), topk_probs[0].item())
        
        # Track active leaves
        active_leaves = [(0, draft_outputs.past_key_values, topk_tokens[0:1])]
        
        for depth in range(1, self.tree_depth + 1):
            if len(tree) >= self.max_tree_nodes or not active_leaves:
                break
            
            new_active_leaves = []
            
            for leaf_idx, leaf_cache, leaf_token in active_leaves:
                leaf_node = tree.nodes[leaf_idx]
                
                # Check cumulative probability threshold
                if leaf_node.cumulative_logit < torch.log(
                    torch.tensor(self.probability_threshold)
                ).item():
                    continue  # Prune this branch
                
                # Forward through draft model
                draft_outputs = self.draft_model(
                    input_ids=leaf_token.unsqueeze(0),
                    past_key_values=leaf_cache,
                    use_cache=True,
                    return_dict=True
                )
                
                next_logits = draft_outputs.logits[:, -1, :]
                
                # Get adaptive branch factor for this node
                node_branch = self._get_adaptive_branch_factor(next_logits[0])
                
                log_probs = F.log_softmax(next_logits, dim=-1)
                topk_probs, topk_tokens = torch.topk(log_probs[0], node_branch)
                
                remaining = self.max_tree_nodes - len(tree)
                num_children = min(node_branch, remaining)
                
                for i in range(num_children):
                    child_idx = tree.add_node(
                        token_id=topk_tokens[i].item(),
                        parent_idx=leaf_idx,
                        logit=topk_probs[i].item()
                    )
                    new_active_leaves.append(
                        (child_idx, draft_outputs.past_key_values, topk_tokens[i:i+1])
                    )
            
            active_leaves = new_active_leaves
        
        self._token_tree = tree
        return tree
    
    def get_stats(self) -> dict:
        """Get generation statistics including adaptive branching stats."""
        stats = super().get_stats()
        stats["adaptive_stats"] = self.adaptive_stats.copy()
        
        # Calculate distribution
        total = self.adaptive_stats["total_expansions"]
        if total > 0:
            stats["adaptive_stats"]["high_conf_ratio"] = self.adaptive_stats["high_conf_count"] / total
            stats["adaptive_stats"]["medium_conf_ratio"] = self.adaptive_stats["medium_conf_count"] / total
            stats["adaptive_stats"]["low_conf_ratio"] = self.adaptive_stats["low_conf_count"] / total
        
        return stats


# =============================================================================
# Phase 2: Dynamic Depth with Early Stopping
# =============================================================================

class TreeSpeculativeGeneratorV2AdaptiveV2(TreeSpeculativeGeneratorV2Adaptive):
    """
    Adaptive Tree Speculative Decoding with Dynamic Depth.
    
    Extends Phase 1 with:
    - Early stopping for low cumulative probability branches
    - Deep expansion for high cumulative probability branches
    
    Args:
        early_stop_threshold: Stop expanding when cumulative prob < threshold (default: 0.1)
        deep_expand_threshold: Allow deeper expansion when cumulative prob > threshold (default: 0.5)
        base_depth: Base tree depth (default: 4)
        max_depth: Maximum allowed depth for high-confidence branches (default: 8)
    """
    
    def __init__(
        self,
        target_model,
        draft_model,
        tokenizer,
        tree_depth: int = 4,  # This becomes base_depth
        branch_factor: int = 2,
        max_tree_nodes: int = 256,
        max_len: int = 2048,
        device: str = "cuda",
        use_compile: bool = True,
        probability_threshold: float = 0.05,
        # Phase 1 parameters
        high_conf_threshold: float = 0.8,
        low_conf_threshold: float = 0.3,
        min_branch: int = 1,
        max_branch: int = 4,
        # Phase 2 parameters
        early_stop_threshold: float = 0.1,
        deep_expand_threshold: float = 0.5,
        max_depth: int = 8,
    ):
        super().__init__(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            tree_depth=tree_depth,
            branch_factor=branch_factor,
            max_tree_nodes=max_tree_nodes,
            max_len=max_len,
            device=device,
            use_compile=use_compile,
            probability_threshold=probability_threshold,
            high_conf_threshold=high_conf_threshold,
            low_conf_threshold=low_conf_threshold,
            min_branch=min_branch,
            max_branch=max_branch
        )
        
        # Phase 2 parameters
        self.base_depth = tree_depth
        self.max_depth = max_depth
        self.early_stop_threshold = early_stop_threshold
        self.deep_expand_threshold = deep_expand_threshold
        
        # Phase 2 statistics
        self.depth_stats = {
            "early_stops": 0,
            "deep_expansions": 0,
            "max_depth_reached": 0,
        }
    
    def reset(self):
        """Reset generator state including depth stats."""
        super().reset()
        self.depth_stats = {
            "early_stops": 0,
            "deep_expansions": 0,
            "max_depth_reached": 0,
        }
    
    def _should_expand(self, node, current_depth: int) -> Tuple[bool, str]:
        """
        Determine whether to expand a node based on cumulative probability and depth.
        
        Args:
            node: TreeNode to evaluate
            current_depth: Current depth in the tree
            
        Returns:
            (should_expand, reason)
        """
        cumulative_prob = math.exp(node.cumulative_logit)
        
        # Early stop for very low probability branches
        if cumulative_prob < self.early_stop_threshold:
            self.depth_stats["early_stops"] += 1
            return False, "early_stop"
        
        # Maximum depth check
        if current_depth >= self.max_depth:
            self.depth_stats["max_depth_reached"] += 1
            return False, "max_depth"
        
        # Beyond base depth, only high-confidence branches continue
        if current_depth >= self.base_depth:
            if cumulative_prob > self.deep_expand_threshold:
                self.depth_stats["deep_expansions"] += 1
                return True, "deep_expand"
            else:
                return False, "base_depth_cutoff"
        
        return True, "normal"
    
    @torch.inference_mode()
    def _draft_tree_tokens(self) -> TokenTree:
        """
        Generate tree with adaptive branching and dynamic depth.
        
        Combines confidence-based branching with:
        - Early stopping for low-probability branches
        - Extended depth for high-probability branches
        """
        tree = TokenTree(
            max_depth=self.max_depth,  # Use max_depth for capacity
            branch_factor=self.max_branch,
            max_nodes=self.max_tree_nodes,
            device=self.device
        )
        
        # Re-prefill draft model
        draft_outputs = self.draft_model(
            input_ids=self.current_ids,
            use_cache=True,
            return_dict=True
        )
        
        first_logits = draft_outputs.logits[:, -1, :]
        root_branch = self._get_adaptive_branch_factor(first_logits[0])
        
        log_probs = F.log_softmax(first_logits, dim=-1)
        topk_probs, topk_tokens = torch.topk(log_probs[0], root_branch)
        
        # Add root
        tree.add_root(topk_tokens[0].item(), topk_probs[0].item())
        
        # Track active leaves with their depth
        active_leaves = [(0, draft_outputs.past_key_values, topk_tokens[0:1], 0)]  # (idx, cache, token, depth)
        
        while active_leaves and len(tree) < self.max_tree_nodes:
            new_active_leaves = []
            
            for leaf_idx, leaf_cache, leaf_token, current_depth in active_leaves:
                leaf_node = tree.nodes[leaf_idx]
                next_depth = current_depth + 1
                
                # Check if we should expand this node
                should_expand, reason = self._should_expand(leaf_node, next_depth)
                if not should_expand:
                    continue
                
                # Also check probability threshold
                if leaf_node.cumulative_logit < torch.log(
                    torch.tensor(self.probability_threshold)
                ).item():
                    continue
                
                # Forward through draft model
                draft_outputs = self.draft_model(
                    input_ids=leaf_token.unsqueeze(0),
                    past_key_values=leaf_cache,
                    use_cache=True,
                    return_dict=True
                )
                
                next_logits = draft_outputs.logits[:, -1, :]
                node_branch = self._get_adaptive_branch_factor(next_logits[0])
                
                log_probs = F.log_softmax(next_logits, dim=-1)
                topk_probs, topk_tokens = torch.topk(log_probs[0], node_branch)
                
                remaining = self.max_tree_nodes - len(tree)
                num_children = min(node_branch, remaining)
                
                for i in range(num_children):
                    try:
                        child_idx = tree.add_node(
                            token_id=topk_tokens[i].item(),
                            parent_idx=leaf_idx,
                            logit=topk_probs[i].item()
                        )
                        new_active_leaves.append(
                            (child_idx, draft_outputs.past_key_values, topk_tokens[i:i+1], next_depth)
                        )
                    except ValueError:
                        # Max depth or max nodes reached
                        break
            
            active_leaves = new_active_leaves
        
        self._token_tree = tree
        return tree
    
    def get_stats(self) -> dict:
        """Get generation statistics including depth stats."""
        stats = super().get_stats()
        stats["depth_stats"] = self.depth_stats.copy()
        stats["base_depth"] = self.base_depth
        stats["max_depth"] = self.max_depth
        return stats


# =============================================================================
# Phase 3: Historical Acceptance Rate Adjustment
# =============================================================================

class TreeSpeculativeGeneratorV2AdaptiveV3(TreeSpeculativeGeneratorV2AdaptiveV2):
    """
    Fully Adaptive Tree Speculative Decoding.
    
    Extends Phase 2 with runtime parameter adjustment based on historical
    acceptance rates:
    - High acceptance rate → More aggressive (deeper trees, higher thresholds)
    - Low acceptance rate → More conservative (shallower trees, lower thresholds)
    
    Args:
        history_window: Number of recent rounds to consider (default: 10)
        target_acceptance_rate: Target acceptance rate to maintain (default: 0.7)
        adjustment_rate: How much to adjust parameters per step (default: 0.05)
        enable_auto_adjust: Whether to enable automatic adjustment (default: True)
    """
    
    def __init__(
        self,
        target_model,
        draft_model,
        tokenizer,
        tree_depth: int = 4,
        branch_factor: int = 2,
        max_tree_nodes: int = 256,
        max_len: int = 2048,
        device: str = "cuda",
        use_compile: bool = True,
        probability_threshold: float = 0.05,
        # Phase 1 parameters
        high_conf_threshold: float = 0.8,
        low_conf_threshold: float = 0.3,
        min_branch: int = 1,
        max_branch: int = 4,
        # Phase 2 parameters
        early_stop_threshold: float = 0.1,
        deep_expand_threshold: float = 0.5,
        max_depth: int = 8,
        # Phase 3 parameters
        history_window: int = 10,
        target_acceptance_rate: float = 0.7,
        adjustment_rate: float = 0.05,
        enable_auto_adjust: bool = True,
    ):
        super().__init__(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            tree_depth=tree_depth,
            branch_factor=branch_factor,
            max_tree_nodes=max_tree_nodes,
            max_len=max_len,
            device=device,
            use_compile=use_compile,
            probability_threshold=probability_threshold,
            high_conf_threshold=high_conf_threshold,
            low_conf_threshold=low_conf_threshold,
            min_branch=min_branch,
            max_branch=max_branch,
            early_stop_threshold=early_stop_threshold,
            deep_expand_threshold=deep_expand_threshold,
            max_depth=max_depth
        )
        
        # Phase 3 parameters
        self.history_window = history_window
        self.target_acceptance_rate = target_acceptance_rate
        self.adjustment_rate = adjustment_rate
        self.enable_auto_adjust = enable_auto_adjust
        
        # History tracking
        self.acceptance_history: List[float] = []
        self.path_length_history: List[float] = []
        
        # Current adjusted parameters (start with initial values)
        self.current_base_depth = tree_depth
        self.current_high_conf_threshold = high_conf_threshold
        self.current_deep_expand_threshold = deep_expand_threshold
        
        # Adjustment history for debugging
        self.adjustment_history: List[dict] = []
    
    def reset(self):
        """Reset generator state but keep history for long-term learning."""
        super().reset()
        # Don't reset history - we want to learn across generations
    
    def full_reset(self):
        """Full reset including history."""
        super().reset()
        self.acceptance_history = []
        self.path_length_history = []
        self.adjustment_history = []
        self.current_base_depth = self.base_depth
        self.current_high_conf_threshold = self.high_conf_threshold
        self.current_deep_expand_threshold = self.deep_expand_threshold
    
    def _update_history(self, acceptance_rate: float, avg_path_length: float):
        """Update acceptance rate history."""
        self.acceptance_history.append(acceptance_rate)
        self.path_length_history.append(avg_path_length)
        
        # Keep only recent history
        if len(self.acceptance_history) > self.history_window:
            self.acceptance_history.pop(0)
        if len(self.path_length_history) > self.history_window:
            self.path_length_history.pop(0)
    
    def _adjust_parameters(self):
        """Adjust parameters based on historical performance."""
        if not self.enable_auto_adjust:
            return
        
        if len(self.acceptance_history) < 5:
            return  # Need enough history
        
        avg_acceptance = sum(self.acceptance_history) / len(self.acceptance_history)
        avg_path_length = sum(self.path_length_history) / len(self.path_length_history)
        
        adjustment = {
            "avg_acceptance": avg_acceptance,
            "avg_path_length": avg_path_length,
            "action": "none",
            "old_base_depth": self.current_base_depth,
            "old_high_conf_threshold": self.current_high_conf_threshold,
        }
        
        if avg_acceptance > self.target_acceptance_rate + 0.1:
            # High acceptance rate - be more aggressive
            # Increase depth, lower threshold (accept more expansion)
            self.current_base_depth = min(self.current_base_depth + 1, self.max_depth - 1)
            self.current_high_conf_threshold = max(
                self.current_high_conf_threshold - self.adjustment_rate, 0.5
            )
            self.current_deep_expand_threshold = max(
                self.current_deep_expand_threshold - self.adjustment_rate, 0.3
            )
            adjustment["action"] = "more_aggressive"
            
        elif avg_acceptance < self.target_acceptance_rate - 0.1:
            # Low acceptance rate - be more conservative
            # Decrease depth, raise threshold
            self.current_base_depth = max(self.current_base_depth - 1, 2)
            self.current_high_conf_threshold = min(
                self.current_high_conf_threshold + self.adjustment_rate, 0.95
            )
            self.current_deep_expand_threshold = min(
                self.current_deep_expand_threshold + self.adjustment_rate, 0.8
            )
            adjustment["action"] = "more_conservative"
        
        adjustment["new_base_depth"] = self.current_base_depth
        adjustment["new_high_conf_threshold"] = self.current_high_conf_threshold
        self.adjustment_history.append(adjustment)
        
        # Apply adjusted parameters
        self.base_depth = self.current_base_depth
        self.high_conf_threshold = self.current_high_conf_threshold
        self.deep_expand_threshold = self.current_deep_expand_threshold
    
    def generate(self, prompt: str, max_new_tokens: int = 100, verbose: bool = False) -> str:
        """
        Generate text with automatic parameter adjustment.
        
        After each generation, parameters are adjusted based on the
        acceptance rate to optimize for the target acceptance rate.
        """
        result = super().generate(prompt, max_new_tokens, verbose)
        
        # Get stats and update history
        stats = self.get_stats()
        acceptance_rate = stats.get("acceptance_rate", 0)
        avg_path_length = stats.get("avg_accepted_path_length", 0)
        
        self._update_history(acceptance_rate, avg_path_length)
        self._adjust_parameters()
        
        return result
    
    def get_stats(self) -> dict:
        """Get generation statistics including adjustment info."""
        stats = super().get_stats()
        
        # Add Phase 3 specific stats
        stats["history_stats"] = {
            "acceptance_history": self.acceptance_history.copy(),
            "path_length_history": self.path_length_history.copy(),
            "history_length": len(self.acceptance_history),
        }
        
        stats["current_params"] = {
            "current_base_depth": self.current_base_depth,
            "current_high_conf_threshold": self.current_high_conf_threshold,
            "current_deep_expand_threshold": self.current_deep_expand_threshold,
        }
        
        if self.adjustment_history:
            stats["last_adjustment"] = self.adjustment_history[-1]
            stats["total_adjustments"] = len(self.adjustment_history)
        
        return stats


__all__ = [
    "TreeSpeculativeGeneratorV2Adaptive",
    "TreeSpeculativeGeneratorV2AdaptiveV2",
    "TreeSpeculativeGeneratorV2AdaptiveV3",
]

