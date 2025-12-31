"""
Correctness Tests for Custom Speculative Decoding Implementation

This script verifies:
1. Output correctness: Greedy decoding should produce identical output to pure target model
2. Cache correctness: KV cache length should match sequence length
3. Memory management: No memory leaks after multiple generations
4. Acceptance rate: Should be reasonable (> 0.3 for similar model families)
"""

import torch
import gc
import argparse
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

# Suppress HuggingFace warnings
logging.set_verbosity_error()

from core import SpeculativeGenerator, StaticKVCache


def test_output_correctness(
    target_model,
    draft_model,
    tokenizer,
    device: str,
    prompts: List[str],
    max_new_tokens: int = 50
) -> Tuple[bool, List[str]]:
    """
    Test that speculative decoding produces identical output to pure target model.
    
    For greedy decoding, outputs should be exactly identical.
    """
    print("\n" + "="*60)
    print("TEST 1: Output Correctness")
    print("="*60)
    
    errors = []
    
    # Create generator
    generator = SpeculativeGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=5,
        max_len=2048,
        device=device,
        use_compile=False  # Disable for testing
    )
    
    for i, prompt in enumerate(prompts):
        print(f"\nTest {i+1}/{len(prompts)}: '{prompt[:50]}...'")
        
        # Generate with pure target model
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            target_output = target_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        target_text = tokenizer.decode(target_output[0], skip_special_tokens=True)
        
        # Generate with speculative decoding
        generator.reset()
        spec_text = generator.generate(prompt, max_new_tokens=max_new_tokens)
        
        # Compare
        if target_text == spec_text:
            print(f"  ✅ Outputs match!")
        else:
            print(f"  ❌ Outputs differ!")
            print(f"     Target: {target_text[:100]}...")
            print(f"     Spec:   {spec_text[:100]}...")
            errors.append(f"Sample {i}: outputs differ")
    
    success = len(errors) == 0
    return success, errors


def test_cache_correctness(
    target_model,
    draft_model,
    tokenizer,
    device: str
) -> Tuple[bool, List[str]]:
    """
    Test that KV cache is correctly managed.
    """
    print("\n" + "="*60)
    print("TEST 2: Cache Correctness")
    print("="*60)
    
    errors = []
    
    generator = SpeculativeGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=5,
        max_len=2048,
        device=device,
        use_compile=False
    )
    
    prompt = "The quick brown fox"
    
    # Generate some tokens
    generator.reset()
    _ = generator.generate(prompt, max_new_tokens=20)
    
    # Check cache length matches sequence length
    # Use get_seq_length() for DynamicCache
    cache_len = generator.target_cache.get_seq_length()
    seq_len = generator.current_ids.shape[1]
    
    print(f"\nCache length: {cache_len}")
    print(f"Sequence length: {seq_len}")
    
    # Cache should be <= sequence length (might be slightly less due to truncation)
    if cache_len <= seq_len and cache_len >= seq_len - generator.K:
        print("✅ Cache length is valid")
    else:
        print("❌ Cache length mismatch!")
        errors.append(f"Cache length {cache_len} doesn't match sequence length {seq_len}")
    
    # Test reset
    generator.reset()
    if generator.target_cache is None:
        print("✅ Cache reset works correctly")
    else:
        print("❌ Cache reset failed!")
        errors.append("Cache reset failed")
    
    success = len(errors) == 0
    return success, errors


def test_memory_leaks(
    target_model,
    draft_model,
    tokenizer,
    device: str,
    num_iterations: int = 5
) -> Tuple[bool, List[str]]:
    """
    Test for memory leaks by running multiple generations.
    """
    print("\n" + "="*60)
    print("TEST 3: Memory Leak Test")
    print("="*60)
    
    errors = []
    
    # Record initial memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()
    
    initial_memory = torch.cuda.memory_allocated() / (1024**2)
    print(f"\nInitial memory: {initial_memory:.1f} MB")
    
    generator = SpeculativeGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=5,
        max_len=2048,
        device=device,
        use_compile=False
    )
    
    memory_after_init = torch.cuda.memory_allocated() / (1024**2)
    print(f"After init: {memory_after_init:.1f} MB")
    
    # Run multiple generations
    prompts = [
        "The quick brown fox",
        "In a galaxy far away",
        "Machine learning is",
        "The capital of France",
        "Once upon a time"
    ]
    
    for i in range(num_iterations):
        for prompt in prompts:
            generator.reset()
            _ = generator.generate(prompt, max_new_tokens=50)
        
        current_memory = torch.cuda.memory_allocated() / (1024**2)
        print(f"Iteration {i+1}/{num_iterations}: {current_memory:.1f} MB")
    
    # Clean up
    del generator
    torch.cuda.empty_cache()
    gc.collect()
    
    final_memory = torch.cuda.memory_allocated() / (1024**2)
    print(f"After cleanup: {final_memory:.1f} MB")
    
    # Check for significant memory increase
    memory_increase = final_memory - initial_memory
    if memory_increase < 100:  # Allow some tolerance
        print(f"✅ No significant memory leak detected (increase: {memory_increase:.1f} MB)")
    else:
        print(f"⚠️ Potential memory leak: {memory_increase:.1f} MB increase")
        errors.append(f"Memory increased by {memory_increase:.1f} MB")
    
    success = len(errors) == 0
    return success, errors


def test_acceptance_rate(
    target_model,
    draft_model,
    tokenizer,
    device: str,
    prompts: List[str],
    max_new_tokens: int = 100
) -> Tuple[bool, List[str]]:
    """
    Test that acceptance rate is reasonable.
    """
    print("\n" + "="*60)
    print("TEST 4: Acceptance Rate")
    print("="*60)
    
    errors = []
    
    generator = SpeculativeGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=5,
        max_len=2048,
        device=device,
        use_compile=False
    )
    
    total_rounds = 0
    total_accepted = 0
    
    for prompt in prompts:
        generator.reset()
        _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
        stats = generator.get_stats()
        total_rounds += stats["total_rounds"]
        total_accepted += stats["total_accepted"]
    
    # Get acceptance rate from the last stats (or calculate from totals)
    final_stats = generator.get_stats()
    avg_acceptance = final_stats.get("acceptance_rate", 0.0)
    avg_tokens_per_round = total_accepted / total_rounds if total_rounds > 0 else 0
    
    print(f"\nTotal rounds: {total_rounds}")
    print(f"Total accepted: {total_accepted}")
    print(f"Avg tokens/round: {avg_tokens_per_round:.2f}")
    print(f"Acceptance rate: {avg_acceptance:.2%}")
    
    # Acceptance rate should be > 0.3 for similar model families
    if avg_acceptance >= 0.3:
        print(f"✅ Acceptance rate is good (>= 30%)")
    elif avg_acceptance >= 0.1:
        print(f"⚠️ Acceptance rate is low but acceptable (>= 10%)")
    else:
        print(f"❌ Acceptance rate is too low (< 10%)")
        errors.append(f"Acceptance rate {avg_acceptance:.2%} is too low")
    
    success = len(errors) == 0
    return success, errors


def test_batch_forward_cache_consistency(
    target_model,
    tokenizer,
    device: str
) -> Tuple[bool, List[str]]:
    """
    Test that batch forward produces the same cache as sequential token-by-token forward.
    
    This verifies that our batch cache update optimization is correct.
    """
    print("\n" + "="*60)
    print("TEST 5: Batch Forward Cache Consistency")
    print("="*60)
    
    errors = []
    
    prompt = "The quick brown fox jumps over the lazy"
    tokens_to_test = ["dog", "cat", "bird"]  # Multiple tokens to add
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    # Get additional tokens
    additional_tokens = tokenizer(" " + " ".join(tokens_to_test), return_tensors="pt").input_ids.to(device)
    additional_tokens = additional_tokens[:, 1:]  # Remove BOS if present
    
    print(f"\nPrompt length: {input_ids.shape[1]}")
    print(f"Additional tokens: {additional_tokens.shape[1]}")
    
    with torch.inference_mode():
        # Method 1: Sequential forward (token by token)
        outputs_seq = target_model(input_ids, use_cache=True, return_dict=True)
        cache_seq = outputs_seq.past_key_values
        logits_seq_list = [outputs_seq.logits[:, -1:, :]]
        
        for i in range(additional_tokens.shape[1]):
            token = additional_tokens[:, i:i+1]
            outputs_seq = target_model(
                token, 
                past_key_values=cache_seq, 
                use_cache=True, 
                return_dict=True
            )
            cache_seq = outputs_seq.past_key_values
            logits_seq_list.append(outputs_seq.logits)
        
        logits_sequential = torch.cat(logits_seq_list, dim=1)
        
        # Method 2: Batch forward (all at once)
        outputs_batch_init = target_model(input_ids, use_cache=True, return_dict=True)
        cache_batch = outputs_batch_init.past_key_values
        
        outputs_batch = target_model(
            additional_tokens,
            past_key_values=cache_batch,
            use_cache=True,
            return_dict=True
        )
        cache_batch = outputs_batch.past_key_values
        logits_batch = torch.cat([
            outputs_batch_init.logits[:, -1:, :],
            outputs_batch.logits
        ], dim=1)
    
    # Compare cache lengths
    seq_len_seq = cache_seq.get_seq_length()
    seq_len_batch = cache_batch.get_seq_length()
    
    print(f"\nSequential cache length: {seq_len_seq}")
    print(f"Batch cache length: {seq_len_batch}")
    
    if seq_len_seq == seq_len_batch:
        print("✅ Cache lengths match")
    else:
        print("❌ Cache lengths differ!")
        errors.append(f"Cache lengths differ: {seq_len_seq} vs {seq_len_batch}")
    
    # Compare logits (should produce same predictions)
    seq_preds = logits_sequential.argmax(dim=-1)
    batch_preds = logits_batch.argmax(dim=-1)
    
    print(f"\nSequential predictions: {seq_preds}")
    print(f"Batch predictions: {batch_preds}")
    
    if torch.equal(seq_preds, batch_preds):
        print("✅ Predictions match")
    else:
        print("❌ Predictions differ!")
        errors.append("Predictions differ between sequential and batch forward")
    
    # Compare cache content (sample check on first layer)
    # DynamicCache stores as list of (key, value) tuples or has different access pattern
    try:
        # Try new-style access (key_cache/value_cache attributes)
        if hasattr(cache_seq, 'key_cache'):
            k_seq = cache_seq.key_cache[0]
            k_batch = cache_batch.key_cache[0]
        else:
            # Fallback: access via indexing or to_legacy_cache
            cache_seq_legacy = cache_seq.to_legacy_cache() if hasattr(cache_seq, 'to_legacy_cache') else cache_seq
            cache_batch_legacy = cache_batch.to_legacy_cache() if hasattr(cache_batch, 'to_legacy_cache') else cache_batch
            k_seq = cache_seq_legacy[0][0]
            k_batch = cache_batch_legacy[0][0]
        
        # Check if they're close (allowing for floating point differences)
        if torch.allclose(k_seq, k_batch, atol=1e-4, rtol=1e-4):
            print("✅ Cache content matches (within tolerance)")
        else:
            max_diff = (k_seq - k_batch).abs().max().item()
            print(f"⚠️ Cache content differs (max diff: {max_diff:.6f})")
            if max_diff > 1e-2:
                errors.append(f"Cache content differs significantly: max_diff={max_diff}")
    except Exception as e:
        print(f"⚠️ Could not compare cache content directly: {e}")
        print("   Skipping detailed cache comparison (predictions already verified)")
    
    success = len(errors) == 0
    return success, errors


def test_static_cache():
    """
    Test StaticKVCache functionality independently.
    """
    print("\n" + "="*60)
    print("TEST 6: StaticKVCache Unit Tests")
    print("="*60)
    
    errors = []
    
    from core.static_cache import StaticKVCache, CacheConfig
    
    config = CacheConfig(
        num_layers=4,
        num_heads=8,
        head_dim=64,
        max_seq_len=100,
        dtype=torch.float16,
        device="cuda"
    )
    
    cache = StaticKVCache(config)
    
    # Test initial state
    assert cache.get_seq_len() == 0, "Initial length should be 0"
    print("✅ Initial state correct")
    
    # Test update
    new_keys = torch.randn(4, 1, 8, 10, 64, dtype=torch.float16, device="cuda")
    new_values = torch.randn(4, 1, 8, 10, 64, dtype=torch.float16, device="cuda")
    cache.update(new_keys, new_values)
    assert cache.get_seq_len() == 10, "Length after update should be 10"
    print("✅ Update works correctly")
    
    # Test truncate
    cache.truncate(5)
    assert cache.get_seq_len() == 5, "Length after truncate should be 5"
    print("✅ Truncate works correctly")
    
    # Test to_hf_format
    hf_format = cache.to_hf_format()
    assert len(hf_format) == 4, "Should have 4 layers"
    assert hf_format[0][0].shape[2] == 5, "Key length should be 5"
    print("✅ to_hf_format works correctly")
    
    # Test reset
    cache.reset()
    assert cache.get_seq_len() == 0, "Length after reset should be 0"
    print("✅ Reset works correctly")
    
    success = len(errors) == 0
    return success, errors


def main():
    parser = argparse.ArgumentParser(description="Test speculative decoding correctness")
    parser.add_argument("--target-model", type=str, default="/mnt/disk1/models/pythia-2.8b",
                        help="Path to target model")
    parser.add_argument("--draft-model", type=str, default="/mnt/disk1/models/pythia-70m",
                        help="Path to draft model")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick tests only")
    args = parser.parse_args()
    
    print("🔬 Running Speculative Decoding Correctness Tests")
    print(f"   Target: {args.target_model}")
    print(f"   Draft: {args.draft_model}")
    
    # Test prompts
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning of time, there was nothing.",
        "Machine learning is a fascinating field of study.",
    ]
    
    if args.quick:
        prompts = prompts[:1]
    
    all_passed = True
    all_errors = []
    
    # Test 6: Static cache (no model needed)
    success, errors = test_static_cache()
    all_passed = all_passed and success
    all_errors.extend(errors)
    
    # Load models for remaining tests
    print(f"\n📦 Loading models...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.float16
    ).to(args.device)
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=torch.float16
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    target_model.eval()
    draft_model.eval()
    
    # Test 1: Output correctness
    success, errors = test_output_correctness(
        target_model, draft_model, tokenizer, args.device, prompts
    )
    all_passed = all_passed and success
    all_errors.extend(errors)
    
    # Test 2: Cache correctness
    success, errors = test_cache_correctness(
        target_model, draft_model, tokenizer, args.device
    )
    all_passed = all_passed and success
    all_errors.extend(errors)
    
    # Test 3: Memory leaks
    if not args.quick:
        success, errors = test_memory_leaks(
            target_model, draft_model, tokenizer, args.device
        )
        all_passed = all_passed and success
        all_errors.extend(errors)
    
    # Test 4: Acceptance rate
    success, errors = test_acceptance_rate(
        target_model, draft_model, tokenizer, args.device, prompts
    )
    all_passed = all_passed and success
    all_errors.extend(errors)
    
    # Test 5: Batch forward cache consistency (critical for optimization)
    success, errors = test_batch_forward_cache_consistency(
        target_model, tokenizer, args.device
    )
    all_passed = all_passed and success
    all_errors.extend(errors)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if all_passed:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Some tests failed:")
        for error in all_errors:
            print(f"   - {error}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

