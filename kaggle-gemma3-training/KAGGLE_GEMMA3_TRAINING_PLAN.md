# Kaggle-Specific Gemma 3 Training Implementation Plan

## Overview
This plan adapts the official Google Gemma PyTorch implementation for pretraining on Kaggle using your 20GB+ multilingual corpus (English, Hindi, Konkani) and trained SentencePiece tokenizer.

## 🗂️ Project Structure
```
kaggle-gemma3-training/
├── gemma/                          # Official Google Gemma implementation (cloned)
│   ├── model.py                   # Main model classes
│   ├── config.py                  # Model configurations  
│   ├── tokenizer.py               # Tokenizer wrapper
│   └── ...
├── training/                      # Custom training implementation
│   ├── train_gemma3.py           # Main training script
│   ├── data_loader.py             # Kaggle-optimized data loading
│   ├── trainer.py                 # Training loop implementation
│   ├── checkpoint_manager.py      # Checkpointing utilities
│   └── memory_optimizer.py       # Memory management for Kaggle
├── configs/                       # Training configurations
│   ├── gemma3_1b_config.py       # 1B model config
│   ├── gemma3_2b_config.py       # 2B model config  
│   └── training_config.py        # Training hyperparameters
├── utils/                         # Utility functions
│   ├── tokenizer_adapter.py      # Your tokenizer integration
│   ├── data_processor.py         # Data preprocessing
│   └── kaggle_utils.py           # Kaggle-specific helpers
└── requirements.txt               # Dependencies
```

## 🚀 Phase 1: Environment Setup (2-3 hours)

### 1.1 Kaggle Notebook Setup
```python
# Cell 1: Install dependencies
!pip install torch==2.1.1 numpy==1.24.4 sentencepiece==0.1.99 absl-py
!pip install accelerate datasets transformers

# Cell 2: Clone official Google Gemma PyTorch
!git clone https://github.com/google/gemma_pytorch.git
%cd gemma_pytorch
!pip install -e .
```

### 1.2 Data Upload Strategy
Given Kaggle's file size limitations:

**Option A: Kaggle Datasets (Recommended)**
```python
# Upload your data as Kaggle datasets:
# 1. multilingual-corpus-part1 (5GB max)
# 2. multilingual-corpus-part2 (5GB max) 
# 3. multilingual-corpus-part3 (5GB max)
# 4. multilingual-corpus-part4 (5GB max)
# 5. trained-sentencepiece-tokenizer (small)

# Access in notebook:
import kaggle
corpus_paths = [
    '/kaggle/input/multilingual-corpus-part1',
    '/kaggle/input/multilingual-corpus-part2', 
    '/kaggle/input/multilingual-corpus-part3',
    '/kaggle/input/multilingual-corpus-part4'
]
tokenizer_path = '/kaggle/input/trained-sentencepiece-tokenizer'
```

**Option B: Progressive Loading**
```python
# Load data in chunks during training to manage memory
def load_data_chunk(chunk_id, corpus_paths):
    """Load one chunk of data at a time"""
    chunk_files = get_chunk_files(chunk_id, corpus_paths)
    return load_and_tokenize_chunk(chunk_files)
```

### 1.3 GPU Verification
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## 🏗️ Phase 2: Custom Training Implementation (6-8 hours)

### 2.1 Tokenizer Integration (`utils/tokenizer_adapter.py`)
```python
import sentencepiece as spm
from gemma.tokenizer import Tokenizer as GemmaTokenizer

class CustomGemmaTokenizer:
    """Adapter to use your trained SentencePiece tokenizer with Gemma"""
    
    def __init__(self, tokenizer_path):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(tokenizer_path)
        
        # Map to Gemma tokenizer interface
        self.vocab_size = self.sp_model.vocab_size()
        self.bos_id = self.sp_model.bos_id()
        self.eos_id = self.sp_model.eos_id()
        self.pad_id = self.sp_model.pad_id()
        
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs"""
        return self.sp_model.encode(text, out_type=int)
        
    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text"""
        return self.sp_model.decode(token_ids)
```

### 2.2 Model Configuration (`configs/gemma3_1b_config.py`)
```python
from gemma.config import GemmaConfig, Architecture, AttentionType

def get_custom_gemma3_1b_config(vocab_size: int, tokenizer_path: str) -> GemmaConfig:
    """Custom 1B Gemma 3 config with your tokenizer"""
    return GemmaConfig(
        dtype='bfloat16',
        architecture=Architecture.GEMMA_3,
        num_hidden_layers=26,
        num_attention_heads=4,
        num_key_value_heads=1,
        hidden_size=1152,
        intermediate_size=6912,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        head_dim=256,
        attn_types=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
        sliding_window_size=512,
        rope_wave_length={
            AttentionType.LOCAL_SLIDING: 10_000,
            AttentionType.GLOBAL: 1_000_000,
        },
        vocab_size=vocab_size,  # Your tokenizer vocab size
        max_position_embeddings=32_768,
        tokenizer=tokenizer_path,  # Your tokenizer path
        use_qk_norm=True,
        vision_config=None,
        rms_norm_eps=1e-6,
    )
```

### 2.3 Data Loader (`training/data_loader.py`)
```python
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random

class MultilingualDataset(Dataset):
    """Kaggle-optimized dataset for multilingual corpus"""
    
    def __init__(self, corpus_paths, tokenizer, max_length=2048, chunk_size=1000):
        self.corpus_paths = corpus_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        
        # Index all available files
        self.file_index = self._build_file_index()
        self.current_chunk = None
        self.current_chunk_idx = -1
        
    def _build_file_index(self):
        """Build index of all corpus files"""
        file_index = []
        for corpus_path in self.corpus_paths:
            for lang_dir in ['english', 'hindi', 'konkani']:
                lang_path = f"{corpus_path}/{lang_dir}"
                if os.path.exists(lang_path):
                    for file in os.listdir(lang_path):
                        if file.endswith('.jsonl'):
                            file_index.append(f"{lang_path}/{file}")
        return file_index
    
    def _load_chunk(self, chunk_idx):
        """Load a chunk of data on-demand"""
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, len(self.file_index))
        
        chunk_data = []
        for file_path in self.file_index[start_idx:end_idx]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        if 'text' in data:
                            chunk_data.append(data['text'])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
                
        return chunk_data
    
    def __len__(self):
        return len(self.file_index) * 100  # Approximate
    
    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        
        # Load new chunk if needed
        if chunk_idx != self.current_chunk_idx:
            self.current_chunk = self._load_chunk(chunk_idx)
            self.current_chunk_idx = chunk_idx
            
        # Get random text from current chunk
        if self.current_chunk:
            text = random.choice(self.current_chunk)
            tokens = self.tokenizer.encode(text)
            
            # Truncate or pad to max_length
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            elif len(tokens) < self.max_length:
                tokens.extend([self.tokenizer.pad_id] * (self.max_length - len(tokens)))
                
            return {
                'input_ids': torch.tensor(tokens, dtype=torch.long),
                'attention_mask': torch.tensor([1 if t != self.tokenizer.pad_id else 0 for t in tokens], dtype=torch.long)
            }
        else:
            # Fallback to empty sequence
            return {
                'input_ids': torch.tensor([self.tokenizer.pad_id] * self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long)
            }

def create_data_loader(corpus_paths, tokenizer, batch_size=2, max_length=2048):
    """Create DataLoader for training"""
    dataset = MultilingualDataset(corpus_paths, tokenizer, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
```

### 2.4 Training Loop (`training/trainer.py`)
```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
from tqdm.auto import tqdm

class GemmaTrainer:
    """Kaggle-optimized Gemma trainer"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate * 0.1
        )
        
        self.step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(self.device)
        
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Forward pass
        # Use causal language modeling loss
        targets = input_ids.clone()
        targets[attention_mask == 0] = -100  # Ignore padding tokens
        
        # Shift targets for next token prediction
        inputs = input_ids[:, :-1]
        targets = targets[:, 1:]
        
        # Get model outputs (logits)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = self.model(inputs)  # This needs to be adapted for training
            
            # Calculate loss
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        self.step += 1
        
        return loss.item()
    
    def train(self, data_loader, max_steps, checkpoint_dir, save_every=1000):
        """Main training loop"""
        print(f"Starting training for {max_steps} steps...")
        
        progress_bar = tqdm(total=max_steps, desc="Training")
        running_loss = 0.0
        
        data_iter = iter(data_loader)
        
        for step in range(max_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                # Reset data loader
                data_iter = iter(data_loader)
                batch = next(data_iter)
            
            # Training step
            loss = self.train_step(batch)
            running_loss += loss
            
            # Logging
            if (step + 1) % 100 == 0:
                avg_loss = running_loss / 100
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'})
                running_loss = 0.0
            
            # Checkpointing
            if (step + 1) % save_every == 0:
                self.save_checkpoint(checkpoint_dir, step + 1)
                
            progress_bar.update(1)
            
            # Early stopping for Kaggle time limits
            if step >= max_steps - 1:
                break
                
        progress_bar.close()
        print("Training completed!")
```

### 2.5 Checkpoint Manager (`training/checkpoint_manager.py`)
```python
import torch
import os
import json

class CheckpointManager:
    """Manage model checkpoints for Kaggle environment"""
    
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def save_checkpoint(self, checkpoint_dir, step, loss=None):
        """Save model checkpoint"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata
        metadata = {
            'step': step,
            'loss': loss,
            'checkpoint_path': checkpoint_path
        }
        
        metadata_path = os.path.join(checkpoint_dir, 'latest_checkpoint.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        print(f"Checkpoint saved: {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['step'], checkpoint.get('loss', None)
```

## 🔧 Phase 3: Memory Optimization for Kaggle (2-3 hours)

### 3.1 Memory Optimizer (`training/memory_optimizer.py`)
```python
import torch
import gc

class KaggleMemoryOptimizer:
    """Memory optimization utilities for Kaggle environment"""
    
    @staticmethod
    def setup_memory_efficient_training():
        """Setup memory efficient training"""
        # Enable memory efficient attention
        torch.backends.cuda.enable_flash_sdp(True)
        
        # Set memory fraction
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)
            
    @staticmethod
    def clear_cache():
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    @staticmethod
    def get_model_size_mb(model):
        """Calculate model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
        
    @staticmethod
    def optimize_batch_size(model, max_length=2048, target_memory_gb=14):
        """Automatically determine optimal batch size"""
        model_size_mb = KaggleMemoryOptimizer.get_model_size_mb(model)
        
        # Estimate memory per sample (rough)
        memory_per_sample_mb = (max_length * 4 * 2) / (1024 * 1024)  # 4 bytes per token, 2 for forward+backward
        
        available_memory_mb = target_memory_gb * 1024 - model_size_mb - 2048  # Reserve 2GB
        max_batch_size = int(available_memory_mb / memory_per_sample_mb)
        
        return max(1, min(max_batch_size, 8))  # Cap at 8 for stability
```

## 🎯 Phase 4: Main Training Script (1 hour)

### 4.1 Main Training Script (`training/train_gemma3.py`)
```python
#!/usr/bin/env python3
"""
Kaggle-optimized Gemma 3 pretraining script
"""

import torch
import os
import argparse
from pathlib import Path

# Import official Gemma components
from gemma.model import GemmaForCausalLM
from gemma.config import get_model_config

# Import custom components
from utils.tokenizer_adapter import CustomGemmaTokenizer
from training.data_loader import create_data_loader
from training.trainer import GemmaTrainer
from training.checkpoint_manager import CheckpointManager
from training.memory_optimizer import KaggleMemoryOptimizer
from configs.gemma3_1b_config import get_custom_gemma3_1b_config

def main():
    parser = argparse.ArgumentParser(description='Train Gemma 3 on Kaggle')
    parser.add_argument('--model_size', type=str, default='1b', choices=['1b', '2b', '4b'])
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--checkpoint_dir', type=str, default='/kaggle/working/checkpoints')
    parser.add_argument('--resume_from', type=str, default=None)
    
    args = parser.parse_args()
    
    print("🚀 Starting Kaggle Gemma 3 Training")
    print(f"Model size: {args.model_size}")
    print(f"Max steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size}")
    
    # Setup memory optimization
    KaggleMemoryOptimizer.setup_memory_efficient_training()
    
    # Load tokenizer
    print("📝 Loading custom tokenizer...")
    tokenizer_path = '/kaggle/input/trained-sentencepiece-tokenizer/multilingual.model'
    tokenizer = CustomGemmaTokenizer(tokenizer_path)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Create model config
    print("🏗️ Creating model configuration...")
    if args.model_size == '1b':
        config = get_custom_gemma3_1b_config(tokenizer.vocab_size, tokenizer_path)
    else:
        raise NotImplementedError(f"Model size {args.model_size} not implemented yet")
    
    # Initialize model
    print("🧠 Initializing model...")
    model = GemmaForCausalLM(config)
    model_size_mb = KaggleMemoryOptimizer.get_model_size_mb(model)
    print(f"Model size: {model_size_mb:.1f} MB")
    
    # Optimize batch size
    optimal_batch_size = KaggleMemoryOptimizer.optimize_batch_size(model, args.max_length)
    batch_size = min(args.batch_size, optimal_batch_size)
    print(f"Using batch size: {batch_size}")
    
    # Setup data loader
    print("📚 Setting up data loader...")
    corpus_paths = [
        '/kaggle/input/multilingual-corpus-part1',
        '/kaggle/input/multilingual-corpus-part2',
        '/kaggle/input/multilingual-corpus-part3',
        '/kaggle/input/multilingual-corpus-part4'
    ]
    data_loader = create_data_loader(corpus_paths, tokenizer, batch_size, args.max_length)
    
    # Setup trainer
    print("🏃 Setting up trainer...")
    training_config = type('Config', (), {
        'learning_rate': args.learning_rate,
        'max_steps': args.max_steps
    })()
    
    trainer = GemmaTrainer(model, tokenizer, training_config)
    
    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(model, trainer.optimizer, trainer.scheduler)
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from:
        print(f"📂 Resuming from checkpoint: {args.resume_from}")
        start_step, _ = checkpoint_manager.load_checkpoint(args.resume_from)
        trainer.step = start_step
    
    # Start training
    print("🎯 Starting training...")
    try:
        trainer.train(data_loader, args.max_steps, args.checkpoint_dir, save_every=1000)
    except KeyboardInterrupt:
        print("⏹️ Training interrupted, saving checkpoint...")
        checkpoint_manager.save_checkpoint(args.checkpoint_dir, trainer.step)
    except Exception as e:
        print(f"❌ Training error: {e}")
        checkpoint_manager.save_checkpoint(args.checkpoint_dir, trainer.step)
        raise
    
    # Final checkpoint
    print("💾 Saving final checkpoint...")
    checkpoint_manager.save_checkpoint(args.checkpoint_dir, trainer.step)
    
    print("✅ Training completed successfully!")

if __name__ == '__main__':
    main()
```

## 📊 Phase 5: Monitoring and Evaluation (1-2 hours)

### 5.1 Training Monitoring
```python
# Add to training loop
import wandb  # Optional: for experiment tracking

def log_metrics(step, loss, learning_rate, gpu_memory=None):
    """Log training metrics"""
    metrics = {
        'step': step,
        'loss': loss,
        'learning_rate': learning_rate,
        'gpu_memory_gb': gpu_memory
    }
    
    print(f"Step {step}: Loss={loss:.4f}, LR={learning_rate:.2e}")
    
    # Optional: Log to Weights & Biases
    # wandb.log(metrics)
```

### 5.2 Model Evaluation
```python
def evaluate_model(model, tokenizer, test_prompts):
    """Quick evaluation during training"""
    model.eval()
    
    with torch.no_grad():
        for prompt in test_prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor([tokens]).to(model.device)
            
            # Generate continuation
            generated = model.generate(
                input_ids, 
                max_length=input_ids.size(1) + 50,
                temperature=0.7
            )
            
            output_text = tokenizer.decode(generated[0].tolist())
            print(f"Prompt: {prompt}")
            print(f"Generated: {output_text}")
            print("-" * 50)
```

## 🎯 Kaggle-Specific Optimizations

### Resource Management
```python
# Monitor GPU memory
def monitor_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")

# Auto-save for Kaggle time limits
import signal
def setup_auto_save(checkpoint_manager, model_step):
    def signal_handler(sig, frame):
        print('Kaggle time limit approaching, saving checkpoint...')
        checkpoint_manager.save_checkpoint('/kaggle/working/emergency_checkpoint', model_step)
        exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
```

### Data Streaming for Large Corpus
```python
# Stream data to handle 20GB+ corpus
def create_streaming_dataloader(corpus_paths, tokenizer, batch_size):
    """Create memory-efficient streaming data loader"""
    
    def data_generator():
        while True:
            for corpus_path in corpus_paths:
                for root, dirs, files in os.walk(corpus_path):
                    for file in files:
                        if file.endswith('.jsonl'):
                            with open(os.path.join(root, file), 'r') as f:
                                for line in f:
                                    try:
                                        data = json.loads(line.strip())
                                        if 'text' in data:
                                            tokens = tokenizer.encode(data['text'])
                                            if len(tokens) > 10:  # Filter very short texts
                                                yield tokens
                                    except:
                                        continue
    
    return data_generator()
```

## 📈 Expected Timeline & Checkpoints

### Training Schedule (Total: ~20 hours on Kaggle GPU)
- **Phase 1** (2-3h): Environment setup, data upload
- **Phase 2** (6-8h): Implementation of training components
- **Phase 3** (2-3h): Memory optimization and testing
- **Phase 4** (1h): Integration and first training run
- **Phase 5** (1-2h): Monitoring setup
- **Training** (8-12h): Actual model pretraining

### Checkpoint Strategy
- Save every 1,000 steps
- Emergency saves on Kaggle interruption
- Final model export to Kaggle datasets

## 🎯 Success Metrics

1. **Training Loss**: Should decrease consistently
2. **Perplexity**: Calculate on validation set
3. **Generated Text Quality**: Qualitative evaluation
4. **Multilingual Performance**: Test on all three languages
5. **Memory Efficiency**: Stay within Kaggle GPU limits

## 📝 Next Steps

1. **Start with Phase 1**: Set up the Kaggle environment
2. **Upload Data**: Split your 20GB corpus into Kaggle datasets
3. **Test Components**: Run each component individually
4. **Begin Training**: Start with small steps to verify everything works
5. **Scale Up**: Increase to full training once stable

This plan leverages the official Google Gemma implementation while adding the necessary training infrastructure for your specific use case on Kaggle. The modular design allows you to test each component independently and adapt as needed.
