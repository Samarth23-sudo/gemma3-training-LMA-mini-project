# Kaggle Gemma 3 Training

🚀 **Train Gemma 3 models from scratch on Kaggle using your custom multilingual corpus and SentencePiece tokenizer**

This repository provides a complete, Kaggle-optimized implementation for pretraining Google's Gemma 3 language models using the official PyTorch implementation, adapted for custom tokenizers and multilingual corpora.

## ✨ Features

- 🏗️ **Official Google Gemma Architecture**: Uses the official Google Gemma PyTorch implementation
- 🌍 **Multilingual Support**: Optimized for English, Hindi, and Konkani corpus training
- 🎯 **Kaggle Optimized**: Memory-efficient implementation designed for Kaggle's GPU constraints
- 💾 **Smart Checkpointing**: Automatic checkpoint management with cleanup and recovery
- 📊 **Memory Management**: Intelligent batch size optimization and memory monitoring
- ⚡ **Performance Optimized**: Mixed precision training, gradient accumulation, and efficient data loading
- 🔧 **Easy Configuration**: Simple command-line interface with sensible defaults

## 🎯 Supported Models

| Model Size | Parameters | GPU Memory* | Recommended Use |
|------------|------------|-------------|-----------------|
| **Gemma 3 1B** | ~1B | ~8-12GB | ✅ Kaggle training, fast iteration |
| **Gemma 3 2B** | ~2B | ~12-16GB | ⚠️ Kaggle training (tight memory) |

*Estimated GPU memory usage including training overhead

## 🚀 Quick Start

### 1. Upload Your Data to Kaggle

Create these datasets in your Kaggle account:

1. **Tokenizer Dataset**: `trained-sentencepiece-tokenizer`
   - Upload your `multilingual.model` and `multilingual.vocab` files

2. **Corpus Datasets**: Split your 20GB+ corpus into parts (≤5GB each)
   - `multilingual-corpus-part1`
   - `multilingual-corpus-part2` 
   - `multilingual-corpus-part3`
   - `multilingual-corpus-part4`

### 2. Create Kaggle Notebook

1. Create a new Kaggle notebook
2. Enable **GPU T4 x2** acceleration
3. Enable **Internet** access

### 3. Setup Environment

```python
# Install dependencies and clone repositories
!pip install torch==2.1.1 numpy==1.24.4 sentencepiece==0.1.99 tqdm accelerate

# Clone official Google Gemma PyTorch
!git clone https://github.com/google/gemma_pytorch.git
%cd gemma_pytorch && pip install -e . && cd ..

# Clone this training repository
!git clone https://github.com/YOUR_USERNAME/kaggle-gemma3-training.git  # Update URL
```

### 4. Quick Test Run

```python
# Verify everything works with a short test
!python kaggle-gemma3-training/train_gemma3_kaggle.py \
    --test_run \
    --model_size 1b \
    --tokenizer_path /kaggle/input/trained-sentencepiece-tokenizer/multilingual.model \
    --corpus_paths /kaggle/input/multilingual-corpus-part1 \
                   /kaggle/input/multilingual-corpus-part2 \
                   /kaggle/input/multilingual-corpus-part3 \
                   /kaggle/input/multilingual-corpus-part4
```

### 5. Full Training

```python
# Start full pretraining (8-12 hours)
!python kaggle-gemma3-training/train_gemma3_kaggle.py \
    --model_size 1b \
    --max_steps 20000 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 \
    --tokenizer_path /kaggle/input/trained-sentencepiece-tokenizer/multilingual.model \
    --corpus_paths /kaggle/input/multilingual-corpus-part1 \
                   /kaggle/input/multilingual-corpus-part2 \
                   /kaggle/input/multilingual-corpus-part3 \
                   /kaggle/input/multilingual-corpus-part4
```

## 📁 Project Structure

```
kaggle-gemma3-training/
├── train_gemma3_kaggle.py       # Main training script
├── utils/
│   ├── tokenizer_adapter.py     # Custom tokenizer integration
│   └── memory_optimizer.py      # Kaggle memory optimization
├── training/
│   ├── data_loader.py          # Efficient multilingual data loading
│   ├── trainer.py              # Training loop with memory management
│   └── checkpoint_manager.py   # Smart checkpoint management
├── configs/
│   └── gemma3_config.py        # Model configurations
├── requirements.txt            # Dependencies
├── setup_kaggle.sh            # Environment setup script
├── QUICK_START.md             # Quick start guide
└── README.md                  # This file
```

## ⚙️ Command Line Options

### Basic Options
- `--model_size`: Model size (`1b`, `2b`) - Default: `1b`
- `--max_steps`: Training steps - Default: `20000`
- `--batch_size`: Batch size - Default: `2`
- `--learning_rate`: Learning rate - Default: `2e-4`

### Data Options
- `--tokenizer_path`: Path to SentencePiece tokenizer
- `--corpus_paths`: Paths to corpus directories (multiple allowed)
- `--max_length`: Maximum sequence length - Default: `2048`

### Memory Optimization
- `--gradient_accumulation_steps`: Gradient accumulation - Default: `16`
- `--use_amp`: Use mixed precision - Default: `True`

### Checkpointing
- `--checkpoint_dir`: Checkpoint directory - Default: `/kaggle/working/checkpoints`
- `--save_every`: Save frequency - Default: `1000`
- `--resume_from`: Resume from checkpoint path

See `python train_gemma3_kaggle.py --help` for all options.

## 🧠 Model Architecture

This implementation uses Google's official Gemma 3 architecture with:

- **Attention**: Mixed local sliding window + global attention
- **Normalization**: RMSNorm with pre/post feedforward normalization  
- **Activation**: SwiGLU
- **Position Encoding**: RoPE (Rotary Position Embedding)
- **Attention Improvements**: Query-Key normalization, attention logit softcapping

### Gemma 3 1B Configuration
```python
num_hidden_layers=26
num_attention_heads=4  
num_key_value_heads=1
hidden_size=1152
intermediate_size=6912
max_position_embeddings=32768
vocab_size=<your_tokenizer_vocab_size>
```

## 💾 Memory Management

### Automatic Optimizations
- ✅ **Flash Attention**: Enabled automatically if available
- ✅ **Mixed Precision**: bfloat16 training with gradient scaling
- ✅ **Gradient Accumulation**: Simulate large batch sizes
- ✅ **Memory Monitoring**: Real-time GPU/RAM usage tracking
- ✅ **Smart Checkpointing**: Automatic cleanup of old checkpoints

### Manual Optimizations
If you encounter out-of-memory errors:

1. **Reduce batch size**:
   ```bash
   --batch_size 1 --gradient_accumulation_steps 32
   ```

2. **Reduce sequence length**:
   ```bash
   --max_length 1024
   ```

3. **Use smaller model**:
   ```bash
   --model_size 1b
   ```

## 📊 Training Monitoring

The training provides real-time monitoring:

```
Step 1000: Loss=3.245, LR=1.5e-04, GPU: 12.3GB/16.0GB, time/step: 2.1s
Step 2000: Loss=2.893, LR=1.8e-04, GPU: 12.3GB/16.0GB, time/step: 2.0s
```

### Key Metrics
- **Loss**: Should decrease from ~4.0 to ~2.5 over training
- **Learning Rate**: Follows warmup + cosine annealing schedule
- **GPU Memory**: Should stay below 14GB on Kaggle
- **Time/Step**: Expect 1-3 seconds per step depending on batch size

## 🎯 Expected Results

### Training Timeline (Kaggle GPU)
- **Setup**: 5-10 minutes
- **Test Run**: 5-10 minutes  
- **Full Training**: 8-12 hours (20K steps)
- **Total Time**: ~12-13 hours

### Quality Metrics
- **Training Loss**: 4.0 → 2.5 (good), 4.0 → 2.0 (excellent)
- **Perplexity**: 50 → 12 (good), 50 → 7 (excellent)  
- **Generation Quality**: Coherent text in all three languages

## 🔧 Troubleshooting

### Common Issues

#### "CUDA out of memory"
```bash
# Solution: Reduce memory usage
--batch_size 1 --gradient_accumulation_steps 32 --max_length 1024
```

#### "No data files found"
- Verify corpus datasets are uploaded correctly
- Check directory structure: `english/`, `hindi/`, `konkani/`
- Ensure `.jsonl` files are in language directories

#### "Tokenizer not found"
- Verify tokenizer dataset is uploaded
- Check the tokenizer path in command

#### "Training very slow"
- Normal for large models - monitor loss decrease
- Each step processes ~65K tokens (batch_size × gradient_accumulation_steps × max_length)

#### "Loss not decreasing"
- Check data quality and tokenizer consistency
- Verify learning rate isn't too high/low
- Ensure sufficient training data

## 📈 Performance Tips

1. **Start Small**: Begin with `--test_run` to verify setup
2. **Monitor Memory**: Watch GPU usage throughout training  
3. **Use Checkpoints**: Save every 1000 steps for safety
4. **Optimize Batch Size**: Use memory optimizer recommendations
5. **Data Quality**: Ensure clean, well-formatted corpus data

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Test your changes thoroughly
4. Submit a pull request with clear description

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

The official Google Gemma implementation is also under Apache License 2.0.

## 🙏 Acknowledgments

- **Google Gemma Team**: For the official PyTorch implementation
- **Kaggle**: For providing accessible GPU resources
- **SentencePiece Team**: For the tokenization library
- **PyTorch Team**: For the deep learning framework

## 📞 Support

For questions and issues:

1. Check the [QUICK_START.md](QUICK_START.md) guide
2. Review common troubleshooting steps above
3. Open an issue on GitHub with:
   - Error messages
   - Full command used
   - System specifications
   - Sample data format

---

**Happy Training!** 🚀 

*Build amazing multilingual language models with Gemma 3 on Kaggle!*
