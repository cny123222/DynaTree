#!/usr/bin/env python3
"""
Check if models are available locally or need to be downloaded.
"""

import os
from pathlib import Path

def check_model(model_name, common_paths):
    """Check if model exists in common locations."""
    print(f"\n{'='*60}")
    print(f"Checking: {model_name}")
    print(f"{'='*60}")
    
    found = False
    for path in common_paths:
        if os.path.exists(path):
            print(f"✓ Found at: {path}")
            # Check if it has config.json
            if os.path.exists(os.path.join(path, "config.json")):
                print(f"  ✓ Valid model (has config.json)")
                found = True
                return path
            else:
                print(f"  ✗ Invalid (missing config.json)")
    
    if not found:
        print(f"✗ Not found locally")
        print(f"  Will download from HuggingFace on first use")
    
    return None

def main():
    print("="*60)
    print("Model Availability Check")
    print("="*60)
    
    # Common model paths to check
    models_to_check = {
        "pythia-2.8b": [
            "/mnt/disk1/models/pythia-2.8b",
            "./models/pythia-2.8b",
            os.path.expanduser("~/.cache/huggingface/hub/models--EleutherAI--pythia-2.8b"),
        ],
        "pythia-70m": [
            "/mnt/disk1/models/pythia-70m",
            "./models/pythia-70m",
            os.path.expanduser("~/.cache/huggingface/hub/models--EleutherAI--pythia-70m"),
        ],
    }
    
    results = {}
    for model_name, paths in models_to_check.items():
        results[model_name] = check_model(model_name, paths)
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    for model_name, path in results.items():
        if path:
            print(f"✓ {model_name}: {path}")
        else:
            print(f"✗ {model_name}: Will use HuggingFace cache")
    
    # Generate config for experiments
    print(f"\n{'='*60}")
    print("Recommended Model Paths for Experiments")
    print(f"{'='*60}")
    
    if results["pythia-2.8b"]:
        print(f"--target-model {results['pythia-2.8b']}")
    else:
        print(f"--target-model EleutherAI/pythia-2.8b")
    
    if results["pythia-70m"]:
        print(f"--draft-model {results['pythia-70m']}")
    else:
        print(f"--draft-model EleutherAI/pythia-70m")
    
    print()

if __name__ == "__main__":
    main()

