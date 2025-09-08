#!/usr/bin/env python3
"""
Test script to validate Kaggle Gemma 3 training setup
Run this to check if everything is configured correctly before starting training
"""

import os
import sys
import json
from pathlib import Path

def test_imports():
    """Test that all required imports work"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
    except ImportError:
        print("  ❌ PyTorch not found - install with: pip install torch")
        return False
    
    try:
        import sentencepiece
        print(f"  ✅ SentencePiece")
    except ImportError:
        print("  ❌ SentencePiece not found - install with: pip install sentencepiece")
        return False
    
    try:
        import tqdm
        print(f"  ✅ tqdm")
    except ImportError:
        print("  ❌ tqdm not found - install with: pip install tqdm")
        return False
    
    try:
        import numpy
        print(f"  ✅ NumPy {numpy.__version__}")
    except ImportError:
        print("  ❌ NumPy not found - install with: pip install numpy")
        return False
    
    # Test official Gemma import
    try:
        from gemma.model import GemmaForCausalLM
        from gemma.config import GemmaConfig
        print("  ✅ Official Gemma package")
        return True
    except ImportError:
        print("  ⚠️ Official Gemma package not found")
        print("     Install with: git clone https://github.com/google/gemma_pytorch.git && cd gemma_pytorch && pip install -e .")
        return False

def test_gpu():
    """Test GPU availability and setup"""
    print("\n🖥️ Testing GPU setup...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("  ❌ CUDA not available - GPU training will not work")
            return False
        
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"  ✅ CUDA available")
        print(f"  ✅ Device count: {device_count}")
        print(f"  ✅ Device name: {device_name}")
        print(f"  ✅ Total memory: {total_memory:.1f}GB")
        
        # Test memory allocation
        test_tensor = torch.randn(1000, 1000, device='cuda')
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"  ✅ Memory allocation test passed ({allocated:.2f}GB allocated)")
        
        # Cleanup
        del test_tensor
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"  ❌ GPU test failed: {e}")
        return False

def test_tokenizer(tokenizer_path):
    """Test tokenizer loading and functionality"""
    print(f"\n📝 Testing tokenizer: {tokenizer_path}")
    
    if not os.path.exists(tokenizer_path):
        print(f"  ❌ Tokenizer file not found: {tokenizer_path}")
        return False
    
    try:
        # Add current directory to path for imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from utils.tokenizer_adapter import CustomGemmaTokenizer
        
        tokenizer = CustomGemmaTokenizer(tokenizer_path)
        
        print(f"  ✅ Tokenizer loaded successfully")
        print(f"  ✅ Vocab size: {tokenizer.vocab_size}")
        print(f"  ✅ Special tokens - BOS: {tokenizer.bos_id}, EOS: {tokenizer.eos_id}, PAD: {tokenizer.pad_id}")
        
        # Test encoding/decoding
        test_texts = [
            "Hello, how are you?",          # English
            "नमस्ते, आप कैसे हैं?",           # Hindi
            "तुमी कसो आसात?"                # Konkani (if supported)
        ]
        
        print("  🧪 Testing encoding/decoding:")
        for i, text in enumerate(test_texts, 1):
            try:
                tokens = tokenizer.encode(text)
                decoded = tokenizer.decode(tokens)
                print(f"    Test {i}: {len(tokens)} tokens ✅")
            except Exception as e:
                print(f"    Test {i}: Failed - {e} ❌")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Tokenizer test failed: {e}")
        return False

def test_corpus_data(corpus_paths):
    """Test corpus data availability and format"""
    print(f"\n📚 Testing corpus data...")
    
    total_files = 0
    valid_paths = 0
    
    for corpus_path in corpus_paths:
        print(f"  Checking: {corpus_path}")
        
        if not os.path.exists(corpus_path):
            print(f"    ❌ Path does not exist")
            continue
        
        path_files = 0
        for lang_dir in ['english', 'hindi', 'konkani']:
            lang_path = os.path.join(corpus_path, lang_dir)
            
            if os.path.exists(lang_path):
                jsonl_files = [f for f in os.listdir(lang_path) if f.endswith('.jsonl')]
                path_files += len(jsonl_files)
                
                if len(jsonl_files) > 0:
                    # Test loading one file
                    test_file = os.path.join(lang_path, jsonl_files[0])
                    try:
                        with open(test_file, 'r', encoding='utf-8') as f:
                            line = f.readline().strip()
                            if line:
                                data = json.loads(line)
                                if 'text' in data:
                                    print(f"    ✅ {lang_dir}: {len(jsonl_files)} files, valid format")
                                else:
                                    print(f"    ⚠️ {lang_dir}: {len(jsonl_files)} files, missing 'text' field")
                            else:
                                print(f"    ⚠️ {lang_dir}: {len(jsonl_files)} files, empty file")
                    except Exception as e:
                        print(f"    ❌ {lang_dir}: Error reading file - {e}")
                else:
                    print(f"    ⚠️ {lang_dir}: No .jsonl files found")
            else:
                print(f"    ⚠️ {lang_dir}: Directory not found")
        
        if path_files > 0:
            print(f"    ✅ Total files: {path_files}")
            total_files += path_files
            valid_paths += 1
        else:
            print(f"    ❌ No valid files found")
    
    print(f"  📊 Summary: {valid_paths}/{len(corpus_paths)} valid paths, {total_files} total files")
    
    if total_files == 0:
        print("  ❌ No corpus data found - upload your data to Kaggle datasets")
        return False
    elif total_files < 10:
        print("  ⚠️ Very few corpus files found - training may not be effective")
        return True
    else:
        print("  ✅ Sufficient corpus data found")
        return True

def test_custom_modules():
    """Test that our custom modules can be imported"""
    print("\n🔧 Testing custom modules...")
    
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from utils.tokenizer_adapter import CustomGemmaTokenizer
        print("  ✅ CustomGemmaTokenizer")
        
        from training.data_loader import create_data_loader
        print("  ✅ Data loader")
        
        from training.trainer import GemmaTrainer
        print("  ✅ Trainer")
        
        from training.checkpoint_manager import CheckpointManager
        print("  ✅ Checkpoint manager")
        
        from configs.gemma3_config import get_custom_gemma3_1b_config
        print("  ✅ Model configs")
        
        from utils.memory_optimizer import KaggleMemoryOptimizer
        print("  ✅ Memory optimizer")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_memory_requirements():
    """Test memory requirements and provide recommendations"""
    print("\n💾 Testing memory requirements...")
    
    try:
        from utils.memory_optimizer import KaggleMemoryOptimizer
        
        # Check GPU info
        KaggleMemoryOptimizer.print_kaggle_gpu_info()
        
        # Check current memory
        KaggleMemoryOptimizer.print_memory_stats()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Memory test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Kaggle Gemma 3 Training Setup Test")
    print("=" * 60)
    
    # Default paths (adjust for your setup)
    tokenizer_path = "/kaggle/input/trained-sentencepiece-tokenizer/multilingual.model"
    corpus_paths = [
        "/kaggle/input/multilingual-corpus-part1",
        "/kaggle/input/multilingual-corpus-part2", 
        "/kaggle/input/multilingual-corpus-part3",
        "/kaggle/input/multilingual-corpus-part4"
    ]
    
    # Override with command line arguments if provided
    if len(sys.argv) > 1:
        tokenizer_path = sys.argv[1]
    if len(sys.argv) > 2:
        corpus_paths = sys.argv[2:]
    
    print(f"Tokenizer path: {tokenizer_path}")
    print(f"Corpus paths: {corpus_paths}")
    print()
    
    # Run tests
    tests = [
        ("Imports", lambda: test_imports()),
        ("GPU", lambda: test_gpu()),
        ("Custom Modules", lambda: test_custom_modules()),
        ("Tokenizer", lambda: test_tokenizer(tokenizer_path)),
        ("Corpus Data", lambda: test_corpus_data(corpus_paths)),
        ("Memory", lambda: test_memory_requirements()),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name} Test:")
        print("-" * 30)
        
        try:
            if test_func():
                print(f"✅ {test_name} test PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🏁 TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! You're ready to start training!")
        print("\nNext steps:")
        print("1. Run a quick test: python train_gemma3_kaggle.py --test_run")
        print("2. Start full training: python train_gemma3_kaggle.py --model_size 1b")
        return True
    else:
        print("❌ Some tests failed. Please fix the issues before training.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install torch sentencepiece tqdm numpy")
        print("- Install Gemma: git clone https://github.com/google/gemma_pytorch.git && cd gemma_pytorch && pip install -e .")
        print("- Upload tokenizer and corpus data to Kaggle datasets")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
