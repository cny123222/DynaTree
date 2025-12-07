"""
修复pg-19数据集加载问题的补丁

由于pg-19数据集较大且加载复杂，这里提供替代方案
"""

from datasets import load_dataset

def load_long_text_datasets():
    """
    加载适合长文本测试的数据集
    
    Returns:
        tuple: (wikitext_dataset, long_text_dataset)
    """
    print("\nLoading test datasets...")
    
    # 1. Load wikitext
    print("Loading wikitext-2-raw-v1...")
    wikitext_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # 2. 尝试加载pg-19，失败则使用替代方案
    print("Loading long-context dataset...")
    long_text_dataset = None
    
    try:
        # 方法1: 尝试使用新的加载方式
        print("  Attempting to load pg-19 (streaming mode)...")
        pg19_dataset = load_dataset("emozilla/pg19", split="test", streaming=True)
        # 取前几个样本
        long_text_dataset = list(pg19_dataset.take(10))
        print("  ✓ Successfully loaded pg-19 dataset")
    except Exception as e1:
        print(f"  × pg-19 loading failed: {e1}")
        
        try:
            # 方法2: 使用BookCorpus替代（较长文本）
            print("  Attempting to load bookcorpus as alternative...")
            bookcorpus = load_dataset("bookcorpus", split="train", streaming=True)
            long_text_dataset = list(bookcorpus.take(10))
            print("  ✓ Successfully loaded bookcorpus dataset")
        except Exception as e2:
            print(f"  × bookcorpus loading failed: {e2}")
            
            try:
                # 方法3: 使用wikitext的训练集（较长样本）
                print("  Using wikitext train split as alternative...")
                wiki_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                # 筛选较长的样本
                long_samples = [s for s in wiki_train if len(s.get('text', '')) > 1000]
                if long_samples:
                    long_text_dataset = long_samples[:10]
                    print(f"  ✓ Found {len(long_text_dataset)} long samples from wikitext")
                else:
                    print("  × No long samples found in wikitext")
            except Exception as e3:
                print(f"  × All alternatives failed: {e3}")
    
    return wikitext_dataset, long_text_dataset


# 使用示例
if __name__ == "__main__":
    wikitext, long_text = load_long_text_datasets()
    
    print("\n" + "="*60)
    print("Dataset Loading Summary:")
    print("="*60)
    print(f"Wikitext samples: {len(wikitext)}")
    if long_text:
        print(f"Long-text samples: {len(long_text)}")
        if isinstance(long_text, list) and len(long_text) > 0:
            sample = long_text[0]
            text = sample.get('text', '') if isinstance(sample, dict) else str(sample)
            print(f"First sample length: {len(text)} characters")
    else:
        print("Long-text dataset: Not available")

