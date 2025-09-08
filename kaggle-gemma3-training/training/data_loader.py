import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import os
from pathlib import Path
from typing import List, Dict, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilingualDataset(Dataset):
    """
    Kaggle-optimized dataset for multilingual corpus
    Loads data in chunks to handle large corpus efficiently
    """
    
    def __init__(self, corpus_paths: List[str], tokenizer, max_length: int = 2048, 
                 chunk_size: int = 1000, min_length: int = 10):
        """
        Initialize the dataset
        
        Args:
            corpus_paths: List of paths to corpus directories
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            chunk_size: Number of texts to load in each chunk
            min_length: Minimum sequence length to include
        """
        self.corpus_paths = corpus_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.min_length = min_length
        
        # Index all available files
        self.file_index = self._build_file_index()
        self.current_chunk = None
        self.current_chunk_idx = -1
        
        logger.info(f"Dataset initialized with {len(self.file_index)} files")
        logger.info(f"Max length: {max_length}, Min length: {min_length}")
        
    def _build_file_index(self) -> List[str]:
        """Build index of all corpus files"""
        file_index = []
        
        for corpus_path in self.corpus_paths:
            if not os.path.exists(corpus_path):
                logger.warning(f"Corpus path does not exist: {corpus_path}")
                continue
                
            # Look for language directories
            for lang_dir in ['english', 'hindi', 'konkani']:
                lang_path = os.path.join(corpus_path, lang_dir)
                
                if os.path.exists(lang_path):
                    # Look for data files
                    for root, dirs, files in os.walk(lang_path):
                        for file in files:
                            if file.endswith('.jsonl'):
                                full_path = os.path.join(root, file)
                                file_index.append(full_path)
                                
        logger.info(f"Found {len(file_index)} corpus files")
        return file_index
    
    def _load_chunk(self, chunk_idx: int) -> List[str]:
        """Load a chunk of data on-demand"""
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, len(self.file_index))
        
        chunk_data = []
        files_processed = 0
        
        for file_path in self.file_index[start_idx:end_idx]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        try:
                            line = line.strip()
                            if not line:
                                continue
                                
                            data = json.loads(line)
                            
                            # Extract text from different possible fields
                            text = None
                            for field in ['text', 'content', 'body', 'article']:
                                if field in data and data[field]:
                                    text = data[field]
                                    break
                            
                            if text and len(text.strip()) >= self.min_length:
                                chunk_data.append(text.strip())
                                
                        except json.JSONDecodeError:
                            logger.warning(f"JSON decode error in {file_path}:{line_num}")
                            continue
                        except Exception as e:
                            logger.warning(f"Error processing line {line_num} in {file_path}: {e}")
                            continue
                            
                files_processed += 1
                
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
                continue
                
        logger.info(f"Loaded chunk {chunk_idx}: {len(chunk_data)} texts from {files_processed} files")
        return chunk_data
    
    def __len__(self):
        # Approximate length - we don't know exact number without loading all files
        return len(self.file_index) * 100  # Rough estimate
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk_idx = idx // self.chunk_size
        
        # Load new chunk if needed
        if chunk_idx != self.current_chunk_idx:
            self.current_chunk = self._load_chunk(chunk_idx)
            self.current_chunk_idx = chunk_idx
            
        # Get random text from current chunk
        if self.current_chunk and len(self.current_chunk) > 0:
            text = random.choice(self.current_chunk)
            
            # Tokenize text
            try:
                tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
                
                # Truncate or pad to max_length
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                elif len(tokens) < self.max_length:
                    tokens.extend([self.tokenizer.pad_id] * (self.max_length - len(tokens)))
                    
                # Create attention mask
                attention_mask = [1 if t != self.tokenizer.pad_id else 0 for t in tokens]
                
                return {
                    'input_ids': torch.tensor(tokens, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                    'labels': torch.tensor(tokens, dtype=torch.long)  # For causal LM
                }
                
            except Exception as e:
                logger.warning(f"Error tokenizing text: {e}")
                # Return empty sequence as fallback
                return self._get_empty_sequence()
        else:
            # Return empty sequence as fallback
            return self._get_empty_sequence()
    
    def _get_empty_sequence(self) -> Dict[str, torch.Tensor]:
        """Return empty sequence as fallback"""
        tokens = [self.tokenizer.pad_id] * self.max_length
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
            'labels': torch.tensor([-100] * self.max_length, dtype=torch.long)  # Ignore in loss
        }

class StreamingMultilingualDataset:
    """
    Streaming dataset for very large corpora
    Yields data without loading everything into memory
    """
    
    def __init__(self, corpus_paths: List[str], tokenizer, max_length: int = 2048):
        self.corpus_paths = corpus_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __iter__(self):
        """Infinite generator of tokenized sequences"""
        while True:
            for corpus_path in self.corpus_paths:
                if not os.path.exists(corpus_path):
                    continue
                    
                for lang_dir in ['english', 'hindi', 'konkani']:
                    lang_path = os.path.join(corpus_path, lang_dir)
                    
                    if not os.path.exists(lang_path):
                        continue
                        
                    for root, dirs, files in os.walk(lang_path):
                        # Shuffle files for better mixing
                        files = [f for f in files if f.endswith('.jsonl')]
                        random.shuffle(files)
                        
                        for file in files:
                            file_path = os.path.join(root, file)
                            
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    lines = f.readlines()
                                    random.shuffle(lines)  # Shuffle lines within file
                                    
                                    for line in lines:
                                        try:
                                            line = line.strip()
                                            if not line:
                                                continue
                                                
                                            data = json.loads(line)
                                            
                                            # Extract text
                                            text = None
                                            for field in ['text', 'content', 'body', 'article']:
                                                if field in data and data[field]:
                                                    text = data[field]
                                                    break
                                            
                                            if text and len(text.strip()) >= 10:
                                                tokens = self.tokenizer.encode(text.strip(), add_bos=True, add_eos=True)
                                                
                                                if len(tokens) >= 10:  # Minimum length check
                                                    # Truncate if too long
                                                    if len(tokens) > self.max_length:
                                                        tokens = tokens[:self.max_length]
                                                    
                                                    # Pad if too short
                                                    elif len(tokens) < self.max_length:
                                                        tokens.extend([self.tokenizer.pad_id] * (self.max_length - len(tokens)))
                                                    
                                                    attention_mask = [1 if t != self.tokenizer.pad_id else 0 for t in tokens]
                                                    
                                                    yield {
                                                        'input_ids': torch.tensor(tokens, dtype=torch.long),
                                                        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                                                        'labels': torch.tensor(tokens, dtype=torch.long)
                                                    }
                                                    
                                        except Exception as e:
                                            continue
                                            
                            except Exception as e:
                                logger.warning(f"Error processing file {file_path}: {e}")
                                continue

def create_data_loader(corpus_paths: List[str], tokenizer, batch_size: int = 2, 
                      max_length: int = 2048, num_workers: int = 2, 
                      streaming: bool = False) -> Union[DataLoader, StreamingMultilingualDataset]:
    """
    Create DataLoader for training
    
    Args:
        corpus_paths: List of corpus directory paths
        tokenizer: Tokenizer instance
        batch_size: Batch size for training
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        streaming: Whether to use streaming dataset
        
    Returns:
        DataLoader or streaming dataset
    """
    logger.info(f"Creating data loader with batch_size={batch_size}, max_length={max_length}")
    
    if streaming:
        logger.info("Using streaming dataset")
        return StreamingMultilingualDataset(corpus_paths, tokenizer, max_length)
    else:
        logger.info("Using regular dataset with DataLoader")
        dataset = MultilingualDataset(corpus_paths, tokenizer, max_length)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True  # Drop incomplete batches
        )

def test_data_loader(corpus_paths: List[str], tokenizer, max_samples: int = 5):
    """Test the data loader with sample data"""
    print("🧪 Testing Data Loader")
    print("=" * 50)
    
    try:
        # Create data loader
        data_loader = create_data_loader(corpus_paths, tokenizer, batch_size=2, max_length=512)
        
        # Test a few batches
        for i, batch in enumerate(data_loader):
            if i >= max_samples:
                break
                
            print(f"Batch {i+1}:")
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            print(f"  Attention mask shape: {batch['attention_mask'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
            
            # Decode first sample
            first_sample = batch['input_ids'][0]
            decoded_text = tokenizer.decode(first_sample)
            print(f"  Sample text: {decoded_text[:100]}...")
            print("-" * 30)
        
        print("✅ Data loader test completed!")
        
    except Exception as e:
        print(f"❌ Error testing data loader: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test the data loader (you'll need to update paths)
    print("Data loader module loaded successfully!")
    print("To test, call test_data_loader() with your corpus paths and tokenizer")
