"""
从本地parquet文件加载pg-19数据集
"""

import pandas as pd
from datasets import Dataset

def load_local_pg19(parquet_path: str):
    """
    从本地parquet文件加载pg-19数据集
    
    Args:
        parquet_path: parquet文件的路径
    
    Returns:
        Dataset: Hugging Face格式的数据集
    """
    print(f"Loading pg-19 from local parquet file: {parquet_path}")
    
    try:
        # 读取parquet文件
        df = pd.read_parquet(parquet_path)
        
        # 转换为Hugging Face Dataset格式
        dataset = Dataset.from_pandas(df)
        
        print(f"✓ Successfully loaded pg-19 dataset")
        print(f"  - Number of samples: {len(dataset)}")
        
        # 显示数据集信息
        if len(dataset) > 0:
            first_sample = dataset[0]
            if 'text' in first_sample:
                text_len = len(first_sample['text'])
                print(f"  - First sample text length: {text_len} characters")
            elif 'short_book_title' in first_sample:
                print(f"  - First sample title: {first_sample['short_book_title']}")
            
            # 显示所有列名
            print(f"  - Columns: {dataset.column_names}")
        
        return dataset
        
    except Exception as e:
        print(f"✗ Failed to load pg-19 from local file: {e}")
        import traceback
        traceback.print_exc()
        return None


# 测试加载
if __name__ == "__main__":
    parquet_path = "/Users/od/Desktop/NLP/CS2602-LLM-Inference-Acceleration/test-00000-of-00001-29a571947c0b5ccc.parquet"
    
    dataset = load_local_pg19(parquet_path)
    
    if dataset:
        print("\n" + "="*60)
        print("Dataset loaded successfully!")
        print("="*60)
        
        # 显示前几个样本的信息
        print("\nFirst 3 samples:")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            if 'text' in sample:
                text = sample['text']
                print(f"\nSample {i}:")
                print(f"  Length: {len(text)} characters")
                print(f"  Preview: {text[:200]}...")
            elif 'short_book_title' in sample:
                print(f"\nSample {i}:")
                print(f"  Title: {sample.get('short_book_title', 'N/A')}")
                print(f"  Publication date: {sample.get('publication_date', 'N/A')}")

