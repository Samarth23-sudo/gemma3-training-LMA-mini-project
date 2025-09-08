#!/bin/bash
"""
Kaggle Setup Script for Gemma 3 Training
Run this in your Kaggle notebook to set up the environment
"""

echo "🔧 Setting up Kaggle environment for Gemma 3 training..."

# Check if running on Kaggle
if [ -d "/kaggle" ]; then
    echo "✅ Running on Kaggle"
    KAGGLE_ENV=true
else
    echo "ℹ️ Not running on Kaggle"
    KAGGLE_ENV=false
fi

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install torch==2.1.1 numpy==1.24.4 sentencepiece==0.1.99 tqdm accelerate

# Clone and install official Google Gemma PyTorch
echo "🔄 Cloning official Google Gemma PyTorch repository..."
if [ ! -d "gemma_pytorch" ]; then
    git clone https://github.com/google/gemma_pytorch.git
else
    echo "   Repository already exists, updating..."
    cd gemma_pytorch && git pull && cd ..
fi

echo "📦 Installing Gemma PyTorch..."
cd gemma_pytorch && pip install -e . && cd ..

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p /kaggle/working/checkpoints
mkdir -p /kaggle/working/logs
mkdir -p /kaggle/working/exports

# Set environment variables for optimization
echo "⚙️ Setting environment variables..."
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0

# Check GPU availability
echo "🖥️ Checking GPU availability..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('No GPU available')
"

# Test tokenizer (if available)
echo "🧪 Testing tokenizer setup..."
TOKENIZER_PATH="/kaggle/input/trained-sentencepiece-tokenizer/multilingual.model"
if [ -f "$TOKENIZER_PATH" ]; then
    echo "✅ Tokenizer found at $TOKENIZER_PATH"
    python -c "
import sentencepiece as smp
sp = smp.SentencePieceProcessor()
sp.load('$TOKENIZER_PATH')
print(f'Tokenizer vocabulary size: {sp.vocab_size()}')
print('Tokenizer test passed!')
"
else
    echo "⚠️ Tokenizer not found at $TOKENIZER_PATH"
    echo "   Please upload your tokenizer to Kaggle datasets"
fi

# Test corpus data (if available)
echo "🗃️ Checking corpus data..."
CORPUS_PATHS=(
    "/kaggle/input/multilingual-corpus-part1"
    "/kaggle/input/multilingual-corpus-part2"
    "/kaggle/input/multilingual-corpus-part3"
    "/kaggle/input/multilingual-corpus-part4"
)

total_files=0
for path in "${CORPUS_PATHS[@]}"; do
    if [ -d "$path" ]; then
        files=$(find "$path" -name "*.jsonl" | wc -l)
        echo "   Found $files .jsonl files in $path"
        total_files=$((total_files + files))
    else
        echo "   ⚠️ Corpus path not found: $path"
    fi
done

echo "   Total corpus files found: $total_files"

if [ $total_files -eq 0 ]; then
    echo "   ⚠️ No corpus files found. Please upload your data to Kaggle datasets"
fi

echo ""
echo "🎯 Setup completed!"
echo ""
echo "Next steps:"
echo "1. Upload your tokenizer and corpus data to Kaggle datasets"
echo "2. Run the training script:"
echo "   python train_gemma3_kaggle.py --model_size 1b --max_steps 10000"
echo ""
echo "For a quick test run:"
echo "   python train_gemma3_kaggle.py --test_run"
echo ""
