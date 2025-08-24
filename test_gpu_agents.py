#!/usr/bin/env python3
"""
Test GPU configuration across all RL agent types.
"""

import sys
import torch
import numpy as np
sys.path.append('src')

from agents.dqn_agent import DQNAgent
from agents.sac_agent import SACAgent

def test_agent_gpu_config():
    """Test GPU configuration for all agent types."""
    print("=" * 60)
    print(" TESTING GPU CONFIGURATION ACROSS ALL AGENTS")
    print("=" * 60)
    
    # Test parameters
    state_size = 100
    action_size = 3
    config_path = "config/config.yaml"
    
    print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name()}")
    print()
    
    # Test DQN Agent
    print("1. Testing DQN Agent GPU Configuration")
    print("-" * 40)
    try:
        dqn_agent = DQNAgent(state_size, action_size, config_path)
        print(f"   DQN Device: {dqn_agent.device}")
        print(f"   Q-Network Device: {next(dqn_agent.q_network.parameters()).device}")
        print(f"   Target Network Device: {next(dqn_agent.target_network.parameters()).device}")
        
        # Test forward pass
        test_state = torch.randn(1, state_size).to(dqn_agent.device)
        with torch.no_grad():
            q_values = dqn_agent.q_network(test_state)
        print(f"   Q-Values Shape: {q_values.shape}")
        print(f"   Q-Values Device: {q_values.device}")
        print("   DQN Agent: PASSED")
        
    except Exception as e:
        print(f"   DQN Agent: FAILED - {e}")
    
    print()
    
    # Test SAC Agent
    print("2. Testing SAC Agent GPU Configuration")
    print("-" * 40)
    try:
        sac_agent = SACAgent(state_size, action_size, config_path)
        print(f"   SAC Device: {sac_agent.device}")
        print(f"   Actor Device: {next(sac_agent.actor.parameters()).device}")
        print(f"   Critic1 Device: {next(sac_agent.critic1.parameters()).device}")
        print(f"   Critic2 Device: {next(sac_agent.critic2.parameters()).device}")
        print(f"   Replay Buffer Device: {sac_agent.memory.device}")
        
        # Test forward pass
        test_state = torch.randn(1, state_size).to(sac_agent.device)
        with torch.no_grad():
            action, _ = sac_agent.actor.sample(test_state)
            q1_value = sac_agent.critic1(test_state, action)
            q2_value = sac_agent.critic2(test_state, action)
        
        print(f"   Action Shape: {action.shape}, Device: {action.device}")
        print(f"   Q1 Value Shape: {q1_value.shape}, Device: {q1_value.device}")
        print(f"   Q2 Value Shape: {q2_value.shape}, Device: {q2_value.device}")
        print("   SAC Agent: PASSED")
        
    except Exception as e:
        print(f"   SAC Agent: FAILED - {e}")
    
    print()
    
    # Test Multi-Agent System (simplified test without imports)
    print("3. Testing Multi-Agent System GPU Configuration")
    print("-" * 40)
    try:
        # Test individual agents that would be used in multi-agent system
        print("   Testing DQN agent for multi-agent use...")
        dqn_for_multi = DQNAgent(state_size, action_size, config_path)
        print(f"   DQN Device: {dqn_for_multi.device}")
        
        print("   Testing SAC agent for multi-agent use...")
        sac_for_multi = SACAgent(state_size, action_size, config_path)
        print(f"   SAC Device: {sac_for_multi.device}")
        
        print("   Multi-Agent components: PASSED")
        print("   Note: Full multi-agent system test requires running train_multi_agent.py")
        
    except Exception as e:
        print(f"   Multi-Agent System: FAILED - {e}")
    
    print()
    
    # Performance comparison test
    print("4. GPU vs CPU Performance Test")
    print("-" * 40)
    if torch.cuda.is_available():
        try:
            # Test tensor operations on GPU vs CPU
            size = 1000
            iterations = 100
            
            # GPU test
            device_gpu = torch.device('cuda')
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            for _ in range(iterations):
                a = torch.randn(size, size, device=device_gpu)
                b = torch.randn(size, size, device=device_gpu)
                c = torch.mm(a, b)
            end_time.record()
            torch.cuda.synchronize()
            gpu_time = start_time.elapsed_time(end_time)
            
            # CPU test
            device_cpu = torch.device('cpu')
            import time
            start_time_cpu = time.time()
            for _ in range(iterations):
                a = torch.randn(size, size, device=device_cpu)
                b = torch.randn(size, size, device=device_cpu)
                c = torch.mm(a, b)
            end_time_cpu = time.time()
            cpu_time = (end_time_cpu - start_time_cpu) * 1000  # Convert to ms
            
            speedup = cpu_time / gpu_time
            print(f"   GPU Time: {gpu_time:.2f} ms")
            print(f"   CPU Time: {cpu_time:.2f} ms")
            print(f"   GPU Speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"   Performance Test: FAILED - {e}")
    else:
        print("   GPU not available - skipping performance test")
    
    print()
    print("=" * 60)
    print(" GPU CONFIGURATION TEST COMPLETE")
    print("=" * 60)

def test_memory_usage():
    """Test GPU memory usage with different batch sizes."""
    if not torch.cuda.is_available():
        print("GPU not available - skipping memory test")
        return
    
    print("\n5. GPU Memory Usage Test")
    print("-" * 40)
    
    device = torch.device('cuda')
    torch.cuda.empty_cache()  # Clear cache
    
    # Test different batch sizes
    batch_sizes = [32, 64, 128, 256, 512]
    state_size = 384  # From your trading environment
    
    for batch_size in batch_sizes:
        try:
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            
            # Simulate DQN training batch
            states = torch.randn(batch_size, state_size, device=device)
            targets = torch.randn(batch_size, 3, device=device)  # 3 actions
            
            # Simple network forward pass
            net = torch.nn.Sequential(
                torch.nn.Linear(state_size, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 3)
            ).to(device)
            
            outputs = net(states)
            loss = torch.nn.MSELoss()(outputs, targets)
            
            peak_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_used = peak_memory - initial_memory
            
            print(f"   Batch Size {batch_size:3d}: {memory_used:.1f} MB")
            
            del states, targets, outputs, loss, net
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   Batch Size {batch_size:3d}: OUT OF MEMORY")
                break
            else:
                raise e

if __name__ == "__main__":
    test_agent_gpu_config()
    test_memory_usage()
    
    print(f"\n Ready for GPU-accelerated training!")
    print("Run: python train_agent.py (DQN)")
    print("Run: python train_multi_agent.py (Multi-Agent)")
    print("Run: python train_agent_with_tracking.py (Enhanced tracking)")
