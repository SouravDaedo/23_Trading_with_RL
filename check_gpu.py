#!/usr/bin/env python3
"""
GPU Availability Checker
Checks if CUDA/GPU is available for PyTorch training.
"""

import torch
import sys
import platform

def check_gpu_availability():
    """Check and display GPU availability information."""
    print("=" * 60)
    print(" GPU AVAILABILITY CHECK")
    print("=" * 60)
    
    # System information
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"PyTorch Version: {torch.__version__}")
    print()
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {'YES' if cuda_available else 'NO'}")
    
    if cuda_available:
        # CUDA details
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print()
        
        # GPU details
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name}")
            print(f"  Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            print(f"  Multi-processors: {gpu_props.multi_processor_count}")
            print()
        
        # Current device
        current_device = torch.cuda.current_device()
        print(f"Current Device: GPU {current_device}")
        print(f"Device Name: {torch.cuda.get_device_name(current_device)}")
        
        # Memory usage
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
        memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        print(f"Memory Allocated: {memory_allocated:.2f} GB")
        print(f"Memory Reserved: {memory_reserved:.2f} GB")
        print(f"Memory Total: {memory_total:.2f} GB")
        print(f"Memory Free: {memory_total - memory_reserved:.2f} GB")
        
    else:
        print("\nCUDA is not available. Possible reasons:")
        print("  1. No NVIDIA GPU installed")
        print("  2. CUDA drivers not installed")
        print("  3. PyTorch not compiled with CUDA support")
        print("  4. CUDA version mismatch")
        
        # Check if MPS (Apple Silicon) is available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("\n MPS (Apple Silicon) Available: YES")
            print("  You can use MPS for GPU acceleration on Apple Silicon")
        else:
            print("\n MPS (Apple Silicon) Available: NO")
    
    print("\n" + "=" * 60)
    return cuda_available

def test_gpu_tensor_operations():
    """Test basic GPU tensor operations."""
    if not torch.cuda.is_available():
        print("  Cannot test GPU operations - CUDA not available")
        return False
    
    print(" TESTING GPU OPERATIONS")
    print("-" * 40)
    
    try:
        # Create tensors on GPU
        device = torch.device('cuda')
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Perform operations
        print("Creating 1000x1000 tensors on GPU...")
        z = torch.mm(x, y)  # Matrix multiplication
        print(" Matrix multiplication successful")
        
        # Neural network layer test
        linear = torch.nn.Linear(1000, 500).to(device)
        output = linear(x)
        print(" Neural network layer forward pass successful")
        
        # Memory cleanup
        del x, y, z, output, linear
        torch.cuda.empty_cache()
        print(" Memory cleanup successful")
        
        print(" All GPU tests passed!")
        return True
        
    except Exception as e:
        print(f" GPU test failed: {e}")
        return False

def get_recommended_settings():
    """Get recommended settings for training."""
    print("\n RECOMMENDED TRAINING SETTINGS")
    print("-" * 40)
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if gpu_memory >= 8:
            batch_size = 128
            memory_note = "High-end GPU"
        elif gpu_memory >= 4:
            batch_size = 64
            memory_note = "Mid-range GPU"
        else:
            batch_size = 32
            memory_note = "Entry-level GPU"
        
        print(f"GPU Memory: {gpu_memory:.1f} GB ({memory_note})")
        print(f"Recommended Batch Size: {batch_size}")
        print(f"Device Setting: 'cuda' or 'auto'")
        
        # Config.yaml snippet
        print(f"\nAdd to config.yaml:")
        print(f"training:")
        print(f"  device: 'cuda'")
        print(f"  batch_size: {batch_size}")
        
    else:
        print("No GPU available - using CPU settings:")
        print("Recommended Batch Size: 32")
        print("Device Setting: 'cpu'")
        
        print(f"\nAdd to config.yaml:")
        print(f"training:")
        print(f"  device: 'cpu'")
        print(f"  batch_size: 32")

if __name__ == "__main__":
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Test GPU operations if available
    if gpu_available:
        print()
        test_gpu_tensor_operations()
    
    # Get recommended settings
    get_recommended_settings()
    
    print(f"\n{'🚀 Ready for GPU training!' if gpu_available else '💻 Ready for CPU training!'}")
