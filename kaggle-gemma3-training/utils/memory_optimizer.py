import torch
import gc
import psutil
import os
from typing import Dict, Optional

class KaggleMemoryOptimizer:
    """
    Memory optimization utilities specifically designed for Kaggle environment
    """
    
    @staticmethod
    def setup_memory_efficient_training():
        """Setup memory efficient training configurations"""
        print("🔧 Setting up memory efficient training...")
        
        # Enable memory efficient attention if available
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            print("  ✅ Flash attention enabled")
        
        # Set memory fraction to leave some headroom
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)
            print("  ✅ GPU memory fraction set to 95%")
            
        # Set environment variables for better memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        print("  ✅ CUDA memory allocator configured")
        
        # Disable Tokenizers parallelism to save memory
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        print("✅ Memory optimization setup complete")
            
    @staticmethod
    def clear_cache():
        """Clear all caches to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get current memory usage statistics"""
        stats = {}
        
        # GPU memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            total_gpu = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            stats.update({
                'gpu_allocated_gb': allocated,
                'gpu_reserved_gb': reserved,
                'gpu_total_gb': total_gpu,
                'gpu_utilization_pct': (allocated / total_gpu) * 100
            })
        
        # System memory
        memory = psutil.virtual_memory()
        stats.update({
            'system_used_gb': memory.used / 1024**3,
            'system_total_gb': memory.total / 1024**3,
            'system_utilization_pct': memory.percent
        })
        
        return stats
    
    @staticmethod
    def print_memory_stats():
        """Print memory statistics in a readable format"""
        stats = KaggleMemoryOptimizer.get_memory_stats()
        
        print("💾 Memory Usage:")
        if 'gpu_allocated_gb' in stats:
            print(f"  GPU: {stats['gpu_allocated_gb']:.1f}GB / {stats['gpu_total_gb']:.1f}GB "
                  f"({stats['gpu_utilization_pct']:.1f}%)")
            print(f"  GPU Reserved: {stats['gpu_reserved_gb']:.1f}GB")
        
        print(f"  System RAM: {stats['system_used_gb']:.1f}GB / {stats['system_total_gb']:.1f}GB "
              f"({stats['system_utilization_pct']:.1f}%)")
    
    @staticmethod
    def get_model_size_mb(model) -> float:
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
    def optimize_batch_size(model, max_length: int = 2048, 
                          target_memory_gb: float = 14.0) -> int:
        """
        Automatically determine optimal batch size for Kaggle GPU
        
        Args:
            model: The model to analyze
            max_length: Maximum sequence length
            target_memory_gb: Target GPU memory usage (14GB for Kaggle with headroom)
            
        Returns:
            Recommended batch size
        """
        print("🔍 Optimizing batch size for Kaggle GPU...")
        
        # Get model size
        model_size_mb = KaggleMemoryOptimizer.get_model_size_mb(model)
        model_size_gb = model_size_mb / 1024
        
        print(f"  Model size: {model_size_gb:.2f}GB")
        
        # Estimate memory per sample (very rough approximation)
        # This includes activations, gradients, and optimizer states
        bytes_per_token = 4  # bfloat16 = 2 bytes, but account for gradients
        tokens_per_sample = max_length
        
        # Memory breakdown per sample:
        # - Activations: ~4 bytes per token per layer (rough estimate)
        # - Gradients: Same as model parameters per sample
        # - Optimizer states: 2x model parameters (AdamW)
        
        # Simplified estimation
        activation_memory_per_sample_gb = (
            tokens_per_sample * bytes_per_token * 20  # 20 layers approx
        ) / 1024**3
        
        # Available memory for activations
        available_memory_gb = target_memory_gb - model_size_gb - 2.0  # 2GB safety margin
        
        if available_memory_gb <= 0:
            print(f"  ⚠️ Model too large for target memory!")
            return 1
        
        # Calculate max batch size
        max_batch_size = int(available_memory_gb / activation_memory_per_sample_gb)
        
        # Conservative limits
        recommended_batch_size = max(1, min(max_batch_size, 8))
        
        print(f"  Available memory: {available_memory_gb:.2f}GB")
        print(f"  Estimated memory per sample: {activation_memory_per_sample_gb*1024:.0f}MB")
        print(f"  Recommended batch size: {recommended_batch_size}")
        
        return recommended_batch_size
    
    @staticmethod
    def check_kaggle_gpu() -> Dict[str, any]:
        """Check Kaggle GPU specifications and availability"""
        info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': 0,
            'device_name': None,
            'total_memory_gb': 0,
            'is_kaggle': os.path.exists('/kaggle')
        }
        
        if torch.cuda.is_available():
            info['device_count'] = torch.cuda.device_count()
            info['device_name'] = torch.cuda.get_device_name(0)
            info['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return info
    
    @staticmethod
    def print_kaggle_gpu_info():
        """Print Kaggle GPU information"""
        info = KaggleMemoryOptimizer.check_kaggle_gpu()
        
        print("🖥️ GPU Information:")
        print(f"  CUDA Available: {info['cuda_available']}")
        
        if info['cuda_available']:
            print(f"  Device Count: {info['device_count']}")
            print(f"  Device Name: {info['device_name']}")
            print(f"  Total Memory: {info['total_memory_gb']:.1f}GB")
            
            if info['is_kaggle']:
                if 'T4' in str(info['device_name']):
                    print("  ✅ Kaggle T4 GPU detected - good for training!")
                elif 'P100' in str(info['device_name']):
                    print("  ✅ Kaggle P100 GPU detected - excellent for training!")
                else:
                    print(f"  ⚠️ Unknown Kaggle GPU: {info['device_name']}")
            
            # Memory recommendations
            if info['total_memory_gb'] >= 15:
                print("  ✅ Sufficient GPU memory for 1B-2B models")
            elif info['total_memory_gb'] >= 10:
                print("  ⚠️ Limited GPU memory - use small batch sizes")
            else:
                print("  ❌ Insufficient GPU memory for large models")
        else:
            print("  ❌ No GPU available - training will be very slow")
        
        print(f"  Running on Kaggle: {info['is_kaggle']}")
    
    @staticmethod
    def emergency_memory_cleanup():
        """Emergency memory cleanup when OOM occurs"""
        print("🚨 Emergency memory cleanup...")
        
        # Clear all caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # Force garbage collection
        gc.collect()
        
        # Print memory stats after cleanup
        KaggleMemoryOptimizer.print_memory_stats()
        
        print("✅ Emergency cleanup complete")

def monitor_memory_usage(func):
    """Decorator to monitor memory usage of a function"""
    def wrapper(*args, **kwargs):
        print(f"📊 Memory before {func.__name__}:")
        KaggleMemoryOptimizer.print_memory_stats()
        
        result = func(*args, **kwargs)
        
        print(f"📊 Memory after {func.__name__}:")
        KaggleMemoryOptimizer.print_memory_stats()
        
        return result
    return wrapper

if __name__ == "__main__":
    # Test the memory optimizer
    print("🧪 Testing Kaggle Memory Optimizer")
    print("=" * 50)
    
    # Setup optimizations
    KaggleMemoryOptimizer.setup_memory_efficient_training()
    
    # Print GPU info
    KaggleMemoryOptimizer.print_kaggle_gpu_info()
    
    # Print current memory stats
    KaggleMemoryOptimizer.print_memory_stats()
    
    print("\n✅ Memory optimizer test complete!")
