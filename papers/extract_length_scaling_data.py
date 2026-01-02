#!/usr/bin/env python3
"""
Extract sequence length scaling data from parameter search results.
This data can be used directly for Figure 3 or Table 4.
"""
import json
import sys

def extract_length_scaling_data(json_path):
    """Extract the optimal configuration data for different token lengths."""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Optimal configurations for each length
    optimal_configs = [
        (100, 7, 3, 0.03),
        (200, 7, 3, 0.03),
        (300, 7, 3, 0.03),
        (500, 8, 3, 0.03),
        (1000, 6, 3, 0.05),
    ]
    
    results = []
    baselines = {}
    
    # Extract data for each configuration
    for tokens, depth, branch, threshold in optimal_configs:
        for result in data['results']:
            if (result['tokens'] == tokens and 
                result['depth'] == depth and 
                result['branch'] == branch and 
                result['threshold'] == threshold):
                
                results.append({
                    'length': tokens,
                    'config': f"D={depth}, B={branch}, τ={threshold}",
                    'depth': depth,
                    'branch': branch,
                    'threshold': threshold,
                    'throughput': result['throughput'],
                    'speedup': result['speedup'],
                    'baseline_throughput': result['baseline_throughput'],
                    'acceptance_rate': result.get('acceptance_rate', 0),
                    'avg_path_length': result.get('avg_path_length', 0)
                })
                baselines[tokens] = result['baseline_throughput']
                break
    
    return results, baselines

def print_results(results):
    """Print results in a nice format."""
    
    print("=" * 80)
    print("Sequence Length Scaling Results")
    print("(Extracted from tree_param_search_20251231_140952.json)")
    print("=" * 80)
    print()
    
    # Table format
    print(f"{'Length':<12} {'Config':<20} {'Baseline':>12} {'Tree V2':>12} {'Speedup':>10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['length']:>6} tok  {r['config']:<20} "
              f"{r['baseline_throughput']:>10.1f} t/s "
              f"{r['throughput']:>10.1f} t/s "
              f"{r['speedup']:>8.2f}x")
    
    print()
    print("=" * 80)
    print("Key Observations:")
    print("=" * 80)
    print("1. Speedup increases from 100 to 500 tokens (1.43x → 1.79x)")
    print("2. Peak performance at 500 tokens (1.79x speedup)")
    print("3. Slight decrease at 1000 tokens (1.71x) - tree depth reduced to D=6")
    print("4. Optimal depth varies with sequence length:")
    print("   - Short (100-300): D=7")
    print("   - Medium (500): D=8")
    print("   - Long (1000): D=6 (to control tree size)")
    print()

def generate_latex_table(results):
    """Generate LaTeX code for the paper."""
    
    print("=" * 80)
    print("LaTeX Table 4: Sequence Length Scaling")
    print("=" * 80)
    print()
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Performance across Different Sequence Lengths}")
    print(r"\label{tab:length_scaling}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Length & Optimal Config & Baseline & Tree V2 & Speedup \\")
    print(r"\midrule")
    
    for r in results:
        config_str = f"D={r['depth']}, B={r['branch']}, τ={r['threshold']}"
        print(f"{r['length']} & {config_str} & "
              f"{r['baseline_throughput']:.1f} & "
              f"{r['throughput']:.1f} & "
              f"{r['speedup']:.2f}× \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()

def save_json(results, output_path):
    """Save results as JSON for plotting."""
    
    output_data = {
        "experiment": "sequence_length_scaling",
        "source": "tree_param_search_20251231_140952.json",
        "description": "Optimal Tree V2 performance at different generation lengths",
        "results": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Results saved to: {output_path}")
    print()

if __name__ == "__main__":
    input_path = "results/tree_param_search_20251231_140952.json"
    output_path = "results/length_scaling_extracted.json"
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    # Extract data
    results, baselines = extract_length_scaling_data(input_path)
    
    # Print results
    print_results(results)
    
    # Generate LaTeX
    generate_latex_table(results)
    
    # Save JSON
    save_json(results, output_path)
    
    print("=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("1. Use length_scaling_extracted.json for Figure 3 plotting")
    print("2. Copy the LaTeX table above into your paper")
    print("3. No need to run Experiment B - data already available!")
    print()

