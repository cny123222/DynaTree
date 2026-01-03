#!/usr/bin/env python3
"""
Tree Attention Mask Visualization

This script generates visualizations similar to SpecInfer Figure 4, showing:
- Left: Tree structure diagram with node labels
- Right: Attention mask heatmap with color-coded regions

Color scheme:
- Blue (#4A90D9): Prefix attention (tokens can attend to prompt)
- Green (#50C878): Ancestor attention (tokens can attend to ancestors in tree)
- White: Blocked (tokens cannot attend)

Usage:
    python papers/visualize_tree_attention.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Tuple, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Tree Building Functions (Simplified for Visualization)
# =============================================================================

class SimpleTokenTree:
    """
    Simplified token tree for visualization purposes.
    
    This creates a deterministic tree structure for consistent visualization,
    without needing actual model inference.
    """
    
    def __init__(self, max_depth: int, branch_factor: int, device: str = "cpu"):
        self.max_depth = max_depth
        self.branch_factor = branch_factor
        self.device = device
        
        self.nodes = []  # List of (token_id, parent_idx, depth)
        self._build_full_tree()
    
    def _build_full_tree(self):
        """Build a full tree up to max_depth with branch_factor branches."""
        # Root node
        self.nodes.append({
            'token_id': 0,
            'parent_idx': -1,
            'depth': 0,
            'children_idx': [],
            'position': 0,
            'label': 't₀'
        })
        
        node_counter = 1
        
        # Build tree level by level
        for depth in range(1, self.max_depth + 1):
            # Get parent nodes at previous depth
            parents = [i for i, n in enumerate(self.nodes) if n['depth'] == depth - 1]
            
            for parent_idx in parents:
                for branch in range(self.branch_factor):
                    # Create subscript label
                    label = f't_{{{node_counter}}}'
                    
                    self.nodes.append({
                        'token_id': node_counter,
                        'parent_idx': parent_idx,
                        'depth': depth,
                        'children_idx': [],
                        'position': len(self.nodes),
                        'label': f't₀' if node_counter == 0 else self._get_subscript_label(node_counter)
                    })
                    
                    self.nodes[parent_idx]['children_idx'].append(len(self.nodes) - 1)
                    node_counter += 1
        
        # Update positions (BFS order)
        self._flatten_bfs()
    
    def _get_subscript_label(self, num: int) -> str:
        """Convert number to subscript label."""
        subscripts = '₀₁₂₃₄₅₆₇₈₉'
        if num < 10:
            return f't{subscripts[num]}'
        else:
            # Handle multi-digit numbers
            result = 't'
            for digit in str(num):
                result += subscripts[int(digit)]
            return result
    
    def _flatten_bfs(self):
        """Flatten tree in BFS order and update positions."""
        position = 0
        for depth in range(self.max_depth + 1):
            for i, node in enumerate(self.nodes):
                if node['depth'] == depth:
                    node['position'] = position
                    position += 1
    
    def get_nodes_at_depth(self, depth: int) -> List[int]:
        """Get node indices at a specific depth."""
        return [i for i, n in enumerate(self.nodes) if n['depth'] == depth]
    
    def get_path_to_root(self, node_idx: int) -> List[int]:
        """Get path from node to root (including both)."""
        path = [node_idx]
        current = node_idx
        while self.nodes[current]['parent_idx'] != -1:
            current = self.nodes[current]['parent_idx']
            path.append(current)
        return path
    
    def build_attention_mask(self, prefix_len: int = 0) -> np.ndarray:
        """
        Build the tree attention mask.
        
        Returns:
            mask: np.ndarray of shape [num_nodes, prefix_len + num_nodes]
                  Values: 0 = blocked, 1 = prefix, 2 = ancestor
        """
        num_nodes = len(self.nodes)
        total_len = prefix_len + num_nodes
        
        # 0 = blocked, 1 = prefix attention, 2 = ancestor attention
        mask = np.zeros((num_nodes, total_len), dtype=np.int32)
        
        # Sort nodes by position for consistent ordering
        sorted_nodes = sorted(range(num_nodes), key=lambda i: self.nodes[i]['position'])
        
        for i, node_idx in enumerate(sorted_nodes):
            # Prefix attention (all nodes can attend to prefix)
            if prefix_len > 0:
                mask[i, :prefix_len] = 1
            
            # Ancestor attention (node can attend to its path to root)
            path = self.get_path_to_root(node_idx)
            for ancestor_idx in path:
                ancestor_pos = self.nodes[ancestor_idx]['position']
                mask[i, prefix_len + ancestor_pos] = 2
        
        return mask
    
    def get_node_positions_for_plot(self) -> Dict[int, Tuple[float, float]]:
        """
        Calculate x, y positions for each node in the tree visualization.
        
        Returns:
            dict mapping node_idx to (x, y) coordinates
        """
        positions = {}
        
        # Calculate width at each depth
        max_width = self.branch_factor ** self.max_depth
        
        for depth in range(self.max_depth + 1):
            nodes_at_depth = self.get_nodes_at_depth(depth)
            num_nodes = len(nodes_at_depth)
            
            # Calculate x positions evenly spaced
            if num_nodes == 1:
                x_positions = [0.5]
            else:
                x_positions = np.linspace(0.1, 0.9, num_nodes)
            
            y = 1.0 - depth / (self.max_depth + 0.5)
            
            for i, node_idx in enumerate(nodes_at_depth):
                positions[node_idx] = (x_positions[i], y)
        
        return positions


# =============================================================================
# Visualization Functions
# =============================================================================

def draw_tree_structure(ax, tree: SimpleTokenTree, title: str = "Tree Structure"):
    """
    Draw the tree structure on the given axes.
    
    Args:
        ax: matplotlib axes
        tree: SimpleTokenTree instance
        title: Title for the subplot
    """
    positions = tree.get_node_positions_for_plot()
    
    # Draw edges first (so nodes appear on top)
    for node_idx, node in enumerate(tree.nodes):
        if node['parent_idx'] != -1:
            parent_pos = positions[node['parent_idx']]
            node_pos = positions[node_idx]
            ax.plot(
                [parent_pos[0], node_pos[0]],
                [parent_pos[1], node_pos[1]],
                'k-', linewidth=1.5, zorder=1
            )
    
    # Draw nodes
    for node_idx, node in enumerate(tree.nodes):
        x, y = positions[node_idx]
        
        # Node circle
        circle = plt.Circle((x, y), 0.03, color='#4A90D9', ec='black', linewidth=1.5, zorder=2)
        ax.add_patch(circle)
        
        # Node label
        label = tree._get_subscript_label(node_idx)
        ax.text(x, y - 0.07, label, ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Configure axes
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)


def draw_attention_heatmap(ax, tree: SimpleTokenTree, prefix_len: int = 5, title: str = "Attention Mask"):
    """
    Draw the attention mask heatmap on the given axes.
    
    Args:
        ax: matplotlib axes
        tree: SimpleTokenTree instance
        prefix_len: Length of prefix (prompt tokens)
        title: Title for the subplot
    """
    mask = tree.build_attention_mask(prefix_len=prefix_len)
    num_nodes = len(tree.nodes)
    
    # Custom colormap: white (blocked), blue (prefix), green (ancestor)
    colors = ['white', '#4A90D9', '#50C878']  # 0=blocked, 1=prefix, 2=ancestor
    cmap = ListedColormap(colors)
    
    # Plot heatmap
    im = ax.imshow(mask, cmap=cmap, aspect='auto', vmin=0, vmax=2)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, prefix_len + num_nodes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_nodes, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Labels
    row_labels = [tree._get_subscript_label(i) for i in range(num_nodes)]
    col_labels = [f'p{i}' for i in range(prefix_len)] + [tree._get_subscript_label(i) for i in range(num_nodes)]
    
    ax.set_yticks(range(num_nodes))
    ax.set_yticklabels(row_labels, fontsize=9)
    
    # Reduce column labels for readability
    ax.set_xticks(range(0, prefix_len + num_nodes, max(1, (prefix_len + num_nodes) // 15)))
    ax.set_xticklabels([col_labels[i] for i in range(0, prefix_len + num_nodes, max(1, (prefix_len + num_nodes) // 15))], 
                       fontsize=8, rotation=45, ha='right')
    
    ax.set_xlabel('Attend to Position', fontsize=11)
    ax.set_ylabel('Query Token', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Add vertical line to separate prefix from tree tokens
    if prefix_len > 0:
        ax.axvline(x=prefix_len - 0.5, color='red', linestyle='--', linewidth=2, label='Prefix boundary')
    
    return im


def create_legend(fig):
    """Create a legend for the attention mask colors."""
    legend_elements = [
        mpatches.Patch(facecolor='#4A90D9', edgecolor='black', label='Prefix (can attend to prompt)'),
        mpatches.Patch(facecolor='#50C878', edgecolor='black', label='Ancestors (can attend to path)'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='Blocked (cannot attend)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11, 
               bbox_to_anchor=(0.5, 0.02), frameon=True)


def visualize_tree_attention(
    depth: int,
    branch: int,
    threshold: float,
    prefix_len: int = 5,
    output_path: str = None
):
    """
    Create a complete visualization for a specific tree configuration.
    
    Args:
        depth: Tree depth (D)
        branch: Branch factor (B)
        threshold: Probability threshold (t)
        prefix_len: Number of prefix tokens to show
        output_path: Path to save the figure
    """
    # Create tree
    tree = SimpleTokenTree(max_depth=depth, branch_factor=branch)
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2, width_ratios=[1, 1.5], wspace=0.3)
    
    # Left: Tree structure
    ax_tree = fig.add_subplot(gs[0])
    draw_tree_structure(
        ax_tree, tree, 
        title=f'Tree Structure\n(D={depth}, B={branch}, t={threshold})'
    )
    
    # Right: Attention mask heatmap
    ax_mask = fig.add_subplot(gs[1])
    im = draw_attention_heatmap(
        ax_mask, tree, prefix_len=prefix_len,
        title=f'Tree Attention Mask\n({len(tree.nodes)} nodes, {prefix_len} prefix tokens)'
    )
    
    # Add legend
    create_legend(fig)
    
    # Main title
    fig.suptitle(
        f'Tree Attention Mask Visualization (SpecInfer Style)\nConfiguration: D={depth}, B={branch}, t={threshold}',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.close()
    
    return tree


def visualize_attention_mask_detail(
    depth: int,
    branch: int,
    threshold: float,
    prefix_len: int = 3,
    output_path: str = None
):
    """
    Create a detailed attention mask visualization with annotations.
    
    This version shows the mask in more detail with cell annotations.
    """
    tree = SimpleTokenTree(max_depth=depth, branch_factor=branch)
    mask = tree.build_attention_mask(prefix_len=prefix_len)
    num_nodes = len(tree.nodes)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Custom colormap
    colors = ['#FFFFFF', '#4A90D9', '#50C878']
    cmap = ListedColormap(colors)
    
    # Plot heatmap
    im = ax.imshow(mask, cmap=cmap, aspect='equal', vmin=0, vmax=2)
    
    # Add cell values and borders
    for i in range(num_nodes):
        for j in range(prefix_len + num_nodes):
            value = mask[i, j]
            if value == 2:  # Ancestor
                text = '✓'
                color = 'white'
            elif value == 1:  # Prefix
                text = '•'
                color = 'white'
            else:  # Blocked
                text = ''
                color = 'gray'
            
            ax.text(j, i, text, ha='center', va='center', 
                   fontsize=10, color=color, fontweight='bold')
    
    # Grid
    ax.set_xticks(np.arange(-0.5, prefix_len + num_nodes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_nodes, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    # Labels
    row_labels = [tree._get_subscript_label(i) for i in range(num_nodes)]
    col_labels = [f'prefix_{i}' for i in range(prefix_len)] + [tree._get_subscript_label(i) for i in range(num_nodes)]
    
    ax.set_yticks(range(num_nodes))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xticks(range(prefix_len + num_nodes))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=45, ha='right')
    
    ax.set_xlabel('Key/Value Position', fontsize=12)
    ax.set_ylabel('Query Token', fontsize=12)
    
    # Vertical line for prefix boundary
    if prefix_len > 0:
        ax.axvline(x=prefix_len - 0.5, color='red', linestyle='--', linewidth=3)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#4A90D9', edgecolor='black', label='Prefix (• = attend to prompt)'),
        mpatches.Patch(facecolor='#50C878', edgecolor='black', label='Ancestors (✓ = attend to tree path)'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='Blocked')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    
    ax.set_title(
        f'Tree Attention Mask Detail\nD={depth}, B={branch}, t={threshold} | {num_nodes} nodes',
        fontsize=14, fontweight='bold', pad=15
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.close()


def main():
    """Generate visualizations for the specified configurations."""
    
    # Create output directory
    os.makedirs("papers/figures", exist_ok=True)
    
    # Configurations to visualize
    configs = [
        {"depth": 7, "branch": 2, "threshold": 0.05},  # D=7 B=2 t=0.05
        {"depth": 5, "branch": 2, "threshold": 0.05},  # D=5 B=2 t=0.05
    ]
    
    print("="*60)
    print("Tree Attention Mask Visualization")
    print("="*60)
    
    for config in configs:
        D, B, t = config['depth'], config['branch'], config['threshold']
        
        print(f"\nGenerating visualization for D={D}, B={B}, t={t}...")
        
        # Main visualization (tree + heatmap side by side)
        output_main = f"papers/figures/tree_attention_mask_D{D}_B{B}.png"
        tree = visualize_tree_attention(
            depth=D, branch=B, threshold=t,
            prefix_len=5,
            output_path=output_main
        )
        print(f"  Tree has {len(tree.nodes)} nodes")
        
        # Detailed visualization (just the heatmap with annotations)
        # Only for smaller trees to keep it readable
        if len(tree.nodes) <= 63:
            output_detail = f"papers/figures/tree_attention_mask_detail_D{D}_B{B}.png"
            visualize_attention_mask_detail(
                depth=D, branch=B, threshold=t,
                prefix_len=3,
                output_path=output_detail
            )
    
    # Also create a small example for explanation (D=3, B=2)
    print("\nGenerating example visualization (D=3, B=2 for illustration)...")
    visualize_tree_attention(
        depth=3, branch=2, threshold=0.05,
        prefix_len=3,
        output_path="papers/figures/tree_attention_mask_example_D3_B2.png"
    )
    visualize_attention_mask_detail(
        depth=3, branch=2, threshold=0.05,
        prefix_len=3,
        output_path="papers/figures/tree_attention_mask_detail_example_D3_B2.png"
    )
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - papers/figures/tree_attention_mask_D7_B2.png")
    print("  - papers/figures/tree_attention_mask_D5_B2.png")
    print("  - papers/figures/tree_attention_mask_detail_D5_B2.png")
    print("  - papers/figures/tree_attention_mask_example_D3_B2.png")
    print("  - papers/figures/tree_attention_mask_detail_example_D3_B2.png")


if __name__ == "__main__":
    main()

