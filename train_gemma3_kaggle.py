#!/usr/bin/env python3
"""
Main training script for Gemma 3 pretraining on Kaggle
Optimized for multilingual corpus and custom tokenizer
"""

import os
import sys
import argparse
import json
import time
import signal
from pathlib import Path
from typing import Optional

import torch
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_kaggle_environment():
    """Setup Kaggle-specific environment"""
    # Check if running on Kaggle
    is_kaggle = os.path.exists('/kaggle')
    
    if is_kaggle:
        logger.info("🔧 Running on Kaggle - applying optimizations")
        
        # Set environment variables for better performance
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid warnings
        os.environ['OMP_NUM_THREADS'] = '1'  # Control CPU threads
        
        # Memory optimization
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)
            torch.backends.cuda.enable_flash_sdp(True)
    
    return is_kaggle

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = {
        'torch': 'PyTorch',
        'sentencepiece': 'SentencePiece',
        'tqdm': 'tqdm',
        'numpy': 'NumPy'
    }
    
    missing_packages = []
    for package, name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(name)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install them with: pip install torch sentencepiece tqdm numpy")
        return False
    
    # Check for official Gemma package
    try:
        import gemma
        logger.info("✅ Official Gemma package found")
        return True
    except ImportError:
        logger.warning("⚠️ Official Gemma package not found")
        logger.warning("Please install it with:")
        logger.warning("git clone https://github.com/google/gemma_pytorch.git")
        logger.warning("cd gemma_pytorch && pip install -e .")
        return False

def setup_signal_handlers(checkpoint_manager, trainer):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(sig, frame):
        logger.info('🛑 Received interrupt signal - saving checkpoint...')
        if hasattr(trainer, 'step') and hasattr(checkpoint_manager, 'save_checkpoint'):
            emergency_path = '/kaggle/working/emergency_checkpoint.pt'
            try:
                checkpoint_manager.save_checkpoint(
                    '/kaggle/working/emergency_checkpoints',
                    trainer.step,
                    loss=None,
                    is_best=False
                )
                logger.info(f'Emergency checkpoint saved')
            except Exception as e:
                logger.error(f'Failed to save emergency checkpoint: {e}')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def validate_corpus_paths(corpus_paths):
    """Validate that corpus paths exist and contain data"""
    valid_paths = []
    total_files = 0
    
    for corpus_path in corpus_paths:
        if not os.path.exists(corpus_path):
            logger.warning(f"Corpus path does not exist: {corpus_path}")
            continue
            
        files_found = 0
        for lang_dir in ['english', 'hindi', 'konkani']:
            lang_path = os.path.join(corpus_path, lang_dir)
            if os.path.exists(lang_path):
                jsonl_files = [f for f in os.listdir(lang_path) if f.endswith('.jsonl')]
                files_found += len(jsonl_files)
        
        if files_found > 0:
            valid_paths.append(corpus_path)
            total_files += files_found
            logger.info(f"Found {files_found} data files in {corpus_path}")
        else:
            logger.warning(f"No .jsonl files found in {corpus_path}")
    
    logger.info(f"Total valid corpus paths: {len(valid_paths)}")
    logger.info(f"Total data files: {total_files}")
    
    return valid_paths

def main():
    parser = argparse.ArgumentParser(description='Train Gemma 3 on Kaggle with multilingual corpus')
    
    # Model configuration
    parser.add_argument('--model_size', type=str, default='200m', 
                       choices=['150m', '200m', '1b', '2b'],
                       help='Model size to train (150m≈150M params, 200m≈200M params, 1b≈1B params, 2b≈2B params)')
    parser.add_argument('--vocab_size', type=int, default=None,
                       help='Vocabulary size (auto-detected from tokenizer if not provided)')
    
    # Training configuration
    parser.add_argument('--max_steps', type=int, default=20000,
                       help='Maximum training steps')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                       help='Warmup steps')
    
    # Data configuration
    parser.add_argument('--corpus_paths', type=str, nargs='+',
                       default=[
                           '/kaggle/input/multilingual-corpus-part1',
                           '/kaggle/input/multilingual-corpus-part2',
                           '/kaggle/input/multilingual-corpus-part3',
                           '/kaggle/input/multilingual-corpus-part4'
                       ],
                       help='Paths to corpus directories')
    parser.add_argument('--tokenizer_path', type=str,
                       default='/kaggle/input/trained-sentencepiece-tokenizer/multilingual.model',
                       help='Path to SentencePiece tokenizer model')
    parser.add_argument('--streaming_data', action='store_true',
                       help='Use streaming data loader for very large corpora')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='/kaggle/working/checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='Save checkpoint every N steps')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='/kaggle/working',
                       help='Output directory')
    parser.add_argument('--log_every', type=int, default=100,
                       help='Log metrics every N steps')
    
    # Advanced options
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
    parser.add_argument('--compile_model', action='store_true',
                       help='Compile model with torch.compile (may not work on Kaggle)')
    parser.add_argument('--test_run', action='store_true',
                       help='Run a quick test with limited steps')
    
    args = parser.parse_args()
    
    # Setup environment
    logger.info("🚀 Starting Kaggle Gemma 3 Training")
    logger.info("=" * 60)
    
    is_kaggle = setup_kaggle_environment()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependency check failed")
        return 1
    
    # Import modules after dependency check
    try:
        from utils.tokenizer_adapter import CustomGemmaTokenizer
        from training.data_loader import create_data_loader
        from training.trainer import GemmaTrainer
        from training.checkpoint_manager import CheckpointManager
        from configs.gemma3_config import (
            get_custom_gemma3_150m_config, get_custom_gemma3_200m_config,
            get_custom_gemma3_1b_config, get_custom_gemma3_2b_config,
            get_training_config, validate_config, estimate_memory_usage
        )
        
        # Try to import official Gemma model
        from gemma.model import GemmaForCausalLM
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please ensure all dependencies are installed correctly")
        return 1
    
    # Validate inputs
    logger.info("🔍 Validating inputs...")
    
    # Check tokenizer
    if not os.path.exists(args.tokenizer_path):
        logger.error(f"Tokenizer not found: {args.tokenizer_path}")
        return 1
    
    # Check corpus paths
    valid_corpus_paths = validate_corpus_paths(args.corpus_paths)
    if not valid_corpus_paths:
        logger.error("No valid corpus paths found")
        return 1
    
    # Load tokenizer
    logger.info("📝 Loading custom tokenizer...")
    try:
        tokenizer = CustomGemmaTokenizer(args.tokenizer_path)
        actual_vocab_size = tokenizer.vocab_size
        logger.info(f"Tokenizer loaded - vocab size: {actual_vocab_size}")
        
        # Test tokenizer
        test_text = "Hello, how are you? नमस्ते, आप कैसे हैं?"
        test_tokens = tokenizer.encode(test_text)
        test_decoded = tokenizer.decode(test_tokens)
        logger.info(f"Tokenizer test - Original: {test_text}")
        logger.info(f"Tokenizer test - Decoded: {test_decoded}")
        
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return 1
    
    # Create model configuration
    logger.info("🏗️ Creating model configuration...")
    
    vocab_size = args.vocab_size if args.vocab_size else actual_vocab_size
    
    try:
        if args.model_size == '150m':
            model_config = get_custom_gemma3_150m_config(vocab_size, args.tokenizer_path)
        elif args.model_size == '200m':
            model_config = get_custom_gemma3_200m_config(vocab_size, args.tokenizer_path)
        elif args.model_size == '1b':
            model_config = get_custom_gemma3_1b_config(vocab_size, args.tokenizer_path)
        elif args.model_size == '2b':
            model_config = get_custom_gemma3_2b_config(vocab_size, args.tokenizer_path)
        else:
            raise ValueError(f"Unsupported model size: {args.model_size}")
        
        # Validate configuration
        validate_config(model_config, actual_vocab_size)
        
        logger.info(f"Model config: {args.model_size} ({model_config.num_hidden_layers} layers, {model_config.hidden_size} hidden size)")
        
    except Exception as e:
        logger.error(f"Failed to create model configuration: {e}")
        return 1
    
    # Memory estimation
    logger.info("💾 Estimating memory usage...")
    memory_est = estimate_memory_usage(model_config, args.batch_size, args.max_length)
    logger.info(f"Estimated model parameters: {memory_est['model_params_millions']:.1f}M")
    logger.info(f"Estimated GPU memory needed: {memory_est['recommended_gpu_memory_gb']:.1f}GB")
    
    if is_kaggle and memory_est['recommended_gpu_memory_gb'] > 14:
        logger.warning("⚠️ Model may not fit in Kaggle GPU memory (16GB)")
        logger.warning("Consider using smaller batch size or model size")
        
        # Auto-adjust batch size
        if memory_est['recommended_gpu_memory_gb'] > 14:
            args.batch_size = max(1, args.batch_size // 2)
            args.gradient_accumulation_steps *= 2
            logger.info(f"Auto-adjusted: batch_size={args.batch_size}, gradient_accumulation_steps={args.gradient_accumulation_steps}")
    
    # Initialize model
    logger.info("🧠 Initializing model...")
    try:
        model = GemmaForCausalLM(model_config)
        
        # Count actual parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")
        
        # Optional: Compile model
        if args.compile_model and hasattr(torch, 'compile'):
            logger.info("🔧 Compiling model...")
            model = torch.compile(model)
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return 1
    
    # Setup training configuration
    logger.info("⚙️ Setting up training configuration...")
    training_config = get_training_config(args.model_size, kaggle_optimized=is_kaggle)
    
    # Override with command line arguments
    training_config.max_steps = args.max_steps
    training_config.batch_size = args.batch_size
    training_config.learning_rate = args.learning_rate
    training_config.max_length = args.max_length
    training_config.warmup_steps = args.warmup_steps
    training_config.gradient_accumulation_steps = args.gradient_accumulation_steps
    training_config.use_amp = args.use_amp
    training_config.save_every = args.save_every
    training_config.log_every = args.log_every
    
    # Test run adjustments
    if args.test_run:
        training_config.max_steps = 100
        training_config.save_every = 50
        training_config.log_every = 10
        logger.info("🧪 Test run mode - limited steps")
    
    # Setup data loader
    logger.info("📚 Setting up data loader...")
    try:
        data_loader = create_data_loader(
            valid_corpus_paths,
            tokenizer,
            batch_size=training_config.batch_size,
            max_length=training_config.max_length,
            num_workers=getattr(training_config, 'num_workers', 2),
            streaming=args.streaming_data
        )
        logger.info("Data loader created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create data loader: {e}")
        return 1
    
    # Setup trainer
    logger.info("🏃 Setting up trainer...")
    try:
        trainer = GemmaTrainer(model, tokenizer, training_config)
        logger.info("Trainer initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to setup trainer: {e}")
        return 1
    
    # Setup checkpoint manager
    logger.info("💾 Setting up checkpoint manager...")
    try:
        checkpoint_manager = CheckpointManager(
            model, trainer.optimizer, trainer.scheduler,
            max_checkpoints=3 if is_kaggle else 5
        )
        
        # Integrate checkpoint manager with trainer
        trainer._save_checkpoint = checkpoint_manager.save_checkpoint
        
    except Exception as e:
        logger.error(f"Failed to setup checkpoint manager: {e}")
        return 1
    
    # Setup signal handlers
    setup_signal_handlers(checkpoint_manager, trainer)
    
    # Resume from checkpoint if specified
    if args.resume_from:
        logger.info(f"📂 Resuming from checkpoint: {args.resume_from}")
        try:
            step, loss = checkpoint_manager.load_checkpoint(args.resume_from)
            trainer.step = step
            logger.info(f"Resumed from step {step} with loss {loss}")
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            return 1
    
    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        config_dict = {
            'args': vars(args),
            'model_config': {
                'architecture': str(model_config.architecture),
                'vocab_size': model_config.vocab_size,
                'num_hidden_layers': model_config.num_hidden_layers,
                'hidden_size': model_config.hidden_size,
                'num_attention_heads': model_config.num_attention_heads,
                'max_position_embeddings': model_config.max_position_embeddings,
            },
            'training_config': {
                'learning_rate': training_config.learning_rate,
                'batch_size': training_config.batch_size,
                'max_steps': training_config.max_steps,
                'warmup_steps': training_config.warmup_steps,
            },
            'memory_estimate': memory_est,
            'total_parameters': total_params,
        }
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Configuration saved to: {config_path}")
    
    # Start training
    logger.info("🎯 Starting training...")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        trainer.train(
            data_loader=data_loader,
            max_steps=training_config.max_steps,
            checkpoint_dir=args.checkpoint_dir,
            save_every=training_config.save_every,
            log_every=training_config.log_every
        )
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time/3600:.2f} hours")
        
        # Final checkpoint and export
        logger.info("💾 Saving final checkpoint...")
        final_checkpoint_path = checkpoint_manager.save_checkpoint(
            args.checkpoint_dir, trainer.step, is_best=True
        )
        
        # Export for Kaggle dataset
        export_path = os.path.join(args.output_dir, 'final_model.pt')
        checkpoint_manager.export_for_kaggle_dataset(
            args.checkpoint_dir, export_path, include_optimizer=False
        )
        
        logger.info("🎉 Training completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save emergency checkpoint
        try:
            emergency_path = os.path.join(args.output_dir, 'emergency_checkpoint.pt')
            checkpoint_manager.save_checkpoint(args.checkpoint_dir, trainer.step)
            logger.info(f"Emergency checkpoint saved to: {emergency_path}")
        except:
            pass
        
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
