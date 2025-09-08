# Quick Start Guide for Kaggle Gemma 3 Training

This guide will help you quickly set up and start training Gemma 3 on Kaggle with your multilingual corpus.

## 📋 Prerequisites

1. **Kaggle Account** with GPU access enabled
2. **Your trained SentencePiece tokenizer** (multilingual.model file)
3. **Your 20GB+ multilingual corpus** split into manageable chunks

## 🚀 Quick Setup (5 minutes)

### Step 1: Create Kaggle Notebook
1. Go to Kaggle and create a new notebook
2. Enable GPU acceleration: Settings → Accelerator → GPU T4 x2
3. Enable internet access: Settings → Internet → On

### Step 2: Upload Data as Kaggle Datasets

Create these datasets in your Kaggle account:

1. **Tokenizer Dataset**:
   ```
   Name: trained-sentencepiece-tokenizer
   Files: multilingual.model, multilingual.vocab (your tokenizer files)
   ```

2. **Corpus Datasets** (your data is already split - see `/kaggle-datasets/UPLOAD_GUIDE.md`):
   ```
   Name: multilingual-corpus-part1 (3.9GB - English files 0-7)
   Name: multilingual-corpus-part2 (3.7GB - English files 8-15) 
   Name: multilingual-corpus-part3 (4.4GB - Hindi files 1-9)
   Name: multilingual-corpus-part4 (4.9GB - Hindi files 10-18 + All Konkani)
   ```
   
   📁 **Ready to upload:** Your split datasets are in `kaggle-datasets/` directory.
   Follow the `UPLOAD_GUIDE.md` for step-by-step Kaggle upload instructions.

### Step 3: Setup Environment in Kaggle Notebook

```python
# Cell 1: Clone this repository and setup environment
!git clone https://github.com/google/gemma_pytorch.git
%cd gemma_pytorch
!pip install -e .
%cd ..

# Install dependencies
!pip install torch==2.1.1 numpy==1.24.4 sentencepiece==0.1.99 tqdm accelerate

# Download our training code
!wget https://github.com/your-repo/kaggle-gemma3-training.zip  # Update with actual URL
!unzip kaggle-gemma3-training.zip
```

### Step 4: Quick Test Run

```python
# Cell 2: Quick test to verify everything works
!python kaggle-gemma3-training/train_gemma3_kaggle.py \
    --test_run \
    --model_size 1b \
    --max_steps 50 \
    --batch_size 1 \
    --tokenizer_path /kaggle/input/trained-sentencepiece-tokenizer/multilingual.model \
    --corpus_paths /kaggle/input/multilingual-corpus-part1 \
                   /kaggle/input/multilingual-corpus-part2 \
                   /kaggle/input/multilingual-corpus-part3 \
                   /kaggle/input/multilingual-corpus-part4
```

### Step 5: Full Training

```python
# Cell 3: Start full training (can run for hours)
!python kaggle-gemma3-training/train_gemma3_kaggle.py \
    --model_size 1b \
    --max_steps 20000 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 \
    --max_length 2048 \
    --save_every 1000 \
    --tokenizer_path /kaggle/input/trained-sentencepiece-tokenizer/multilingual.model \
    --corpus_paths /kaggle/input/multilingual-corpus-part1 \
                   /kaggle/input/multilingual-corpus-part2 \
                   /kaggle/input/multilingual-corpus-part3 \
                   /kaggle/input/multilingual-corpus-part4 \
    --checkpoint_dir /kaggle/working/checkpoints \
    --output_dir /kaggle/working
```

## 🎯 Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Setup | 5-10 min | Environment setup, data upload |
| Test Run | 5-10 min | Quick validation that everything works |
| Full Training | 8-12 hours | Actual model pretraining |
| Export | 5 min | Save final model for download |

## 📊 Monitoring Progress

The training will display:
- Current loss
- Learning rate
- Training speed (steps/sec)
- GPU memory usage
- Estimated time remaining

Example output:
```
Step 1000: Loss=3.2451, LR=1.5e-04, GPU: 12.3GB/16.0GB
Step 2000: Loss=2.8932, LR=1.8e-04, GPU: 12.3GB/16.0GB
```

## 💾 Checkpoints & Export

Checkpoints are automatically saved every 1000 steps to `/kaggle/working/checkpoints/`

To download your trained model:
1. Final model will be saved as `/kaggle/working/final_model.pt`
2. Download this file from the Kaggle notebook output
3. You can also create a Kaggle dataset from the output for easy sharing

## ⚙️ Memory Optimization Tips

If you run out of GPU memory:

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
   --model_size 1b  # instead of 2b
   ```

## 🔧 Troubleshooting

### "CUDA out of memory"
- Reduce `--batch_size` to 1
- Reduce `--max_length` to 1024
- Use `--model_size 1b`

### "No data files found"
- Check that your corpus datasets are properly uploaded
- Verify the corpus directory structure contains `english/`, `hindi/`, `konkani/` folders
- Ensure `.jsonl` files are in the language folders

### "Tokenizer not found"
- Verify tokenizer dataset is uploaded and accessible
- Check the tokenizer path in the command

### "Training very slow"
- This is normal for large models and datasets
- Each step processes `batch_size * gradient_accumulation_steps * max_length` tokens
- Monitor the loss - it should decrease over time

## 📈 Success Metrics

Your training is successful if:
- ✅ Loss decreases consistently (e.g., from 4.0 to 2.5)
- ✅ Generated text becomes more coherent over time
- ✅ Model can generate text in all three languages
- ✅ No frequent OOM (out of memory) errors

## 🎉 Next Steps After Training

1. **Test your model** with sample prompts in all languages
2. **Fine-tune** for specific tasks if needed
3. **Create a Kaggle dataset** with your trained model
4. **Share your results** with the community!

## 💡 Tips for Success

1. **Start with a test run** to verify everything works
2. **Monitor GPU memory** usage throughout training
3. **Save checkpoints frequently** (every 1000 steps)
4. **Use gradient accumulation** to simulate larger batch sizes
5. **Be patient** - training takes 8-12 hours for good results

Happy training! 🚀
