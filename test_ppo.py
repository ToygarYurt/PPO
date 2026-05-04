"""
Quick Test Script for PPO Implementation
========================================
This script verifies that the PPO implementation works correctly
on a simple environment with minimal training time.

Run this first to verify everything is set up correctly!
"""

import torch
import gymnasium as gym
import numpy as np
from ppo import PPO, Memory


def test_ppo_basics():
    """Test basic PPO functionality."""
    print("\n" + "="*60)
    print("TEST 1: Basic PPO Initialization")
    print("="*60)
    
    try:
        ppo = PPO(
            state_dim=4,
            action_dim=2,
            continuous=False,
            lr=2e-4,
            gamma=0.99,
            eps_clip=0.2,
            K_epochs=4,
            batch_size=32,
            gae_lambda=0.95,
            value_coef=0.5,
            entropy_coef=0.001,
        )
        print("PASS: PPO initialized successfully")
        print(f"  Device: {ppo.device}")
        print(f"  Policy parameters: {sum(p.numel() for p in ppo.policy.parameters())}")
        return True
    except Exception as e:
        print(f"FAIL: Error: {e}")
        return False


def test_action_selection():
    """Test action selection from policy."""
    print("\n" + "="*60)
    print("TEST 2: Action Selection")
    print("="*60)
    
    try:
        ppo = PPO(state_dim=4, action_dim=2, continuous=False)
        
        # Test discrete action
        state = np.array([0.1, -0.5, 0.2, -0.3], dtype=np.float32)
        action, log_prob, value = ppo.select_action(state)
        
        assert isinstance(action, (int, np.integer)), "Action should be integer"
        assert isinstance(log_prob, (float, np.floating)), "Log prob should be float"
        assert isinstance(value, (float, np.floating)), "Value should be float"
        
        print("PASS: Discrete action selection works")
        print(f"  State shape: {state.shape}")
        print(f"  Action: {action}, Log Prob: {log_prob:.4f}, Value: {value:.4f}")
        return True
    except Exception as e:
        print(f"FAIL: Error: {e}")
        return False


def test_continuous_actions():
    """Test continuous action space."""
    print("\n" + "="*60)
    print("TEST 3: Continuous Action Space")
    print("="*60)
    
    try:
        ppo = PPO(state_dim=3, action_dim=1, continuous=True)
        
        state = np.array([0.1, -0.5, 0.2], dtype=np.float32)
        action, log_prob, value = ppo.select_action(state)
        
        assert isinstance(action, np.ndarray), "Action should be numpy array"
        assert action.shape == (1,), f"Action shape should be (1,), got {action.shape}"
        
        print("PASS: Continuous action selection works")
        print(f"  State shape: {state.shape}")
        print(f"  Action shape: {action.shape}")
        print(f"  Action value: {action[0]:.4f}")
        return True
    except Exception as e:
        print(f"FAIL: Error: {e}")
        return False


def test_memory():
    """Test memory buffer functionality."""
    print("\n" + "="*60)
    print("TEST 4: Memory Buffer")
    print("="*60)
    
    try:
        memory = Memory()
        
        # Add some transitions
        for i in range(10):
            memory.states.append(torch.randn(4))
            memory.actions.append(torch.tensor([i % 2]))
            memory.logprobs.append(torch.tensor([-0.5]))
            memory.rewards.append(float(i))
            memory.values.append(0.5)
            memory.is_terminals.append(i % 3 == 2)
        
        print("PASS: Memory buffer works")
        print(f"  Stored transitions: {len(memory.rewards)}")
        print(f"  Memory size: states={len(memory.states)}, "
              f"actions={len(memory.actions)}, rewards={len(memory.rewards)}")
        
        # Test clear
        memory.clear_memory()
        assert len(memory.rewards) == 0, "Memory not cleared"
        print("PASS: Memory cleared successfully")
        
        return True
    except Exception as e:
        print(f"FAIL: Error: {e}")
        return False


def test_environment_integration():
    """Test integration with CartPole environment."""
    print("\n" + "="*60)
    print("TEST 5: Environment Integration (CartPole-v1)")
    print("="*60)
    
    try:
        env = gym.make('CartPole-v1')
        ppo = PPO(state_dim=4, action_dim=2, continuous=False)
        memory = Memory()
        
        state, _ = env.reset()
        
        # Collect one episode
        for step in range(100):
            action, log_prob, value = ppo.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            memory.states.append(torch.tensor(state, dtype=torch.float32))
            memory.actions.append(torch.tensor([action], dtype=torch.int64))
            memory.logprobs.append(torch.tensor([log_prob], dtype=torch.float32))
            memory.rewards.append(reward)
            memory.values.append(value)
            memory.is_terminals.append(done)
            
            state = next_state
            if done:
                break
        
        env.close()
        
        print("PASS: Environment integration works")
        print(f"  Episode collected: {len(memory.rewards)} steps")
        print(f"  Episode reward: {sum(memory.rewards):.1f}")
        return True
    except Exception as e:
        print(f"FAIL: Error: {e}")
        return False


def test_ppo_update():
    """Test PPO update mechanism."""
    print("\n" + "="*60)
    print("TEST 6: PPO Update Mechanism")
    print("="*60)
    
    try:
        ppo = PPO(
            state_dim=4,
            action_dim=2,
            continuous=False,
            K_epochs=1,
            batch_size=32,
        )
        memory = Memory()
        
        # Collect some data
        env = gym.make('CartPole-v1')
        state, _ = env.reset()
        
        for _ in range(200):
            action, log_prob, value = ppo.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            memory.states.append(torch.tensor(state, dtype=torch.float32))
            memory.actions.append(torch.tensor([action], dtype=torch.int64))
            memory.logprobs.append(torch.tensor([log_prob], dtype=torch.float32))
            memory.rewards.append(reward)
            memory.values.append(value)
            memory.is_terminals.append(done)
            
            state = next_state
            if done:
                state, _ = env.reset()
        
        env.close()
        
        # Perform update
        print(f"  Data collected: {len(memory.rewards)} transitions")
        ppo.update(memory, current_episode=1, max_episodes=100)
        print("PASS: PPO update works")
        print("  Policy and value networks updated successfully")
        return True
    except Exception as e:
        print(f"FAIL: Error: {e}")
        return False


def test_device_handling():
    """Test GPU/CPU device handling."""
    print("\n" + "="*60)
    print("TEST 7: Device Handling")
    print("="*60)
    
    try:
        ppo = PPO(state_dim=4, action_dim=2)
        
        print(f"PASS: Device handling works")
        print(f"  Current device: {ppo.device}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  GPU name: {torch.cuda.get_device_name(0)}")
        
        return True
    except Exception as e:
        print(f"FAIL: Error: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("PPO IMPLEMENTATION VERIFICATION TESTS")
    print("="*60)
    
    tests = [
        ("Basic Initialization", test_ppo_basics),
        ("Action Selection", test_action_selection),
        ("Continuous Actions", test_continuous_actions),
        ("Memory Buffer", test_memory),
        ("Environment Integration", test_environment_integration),
        ("PPO Update", test_ppo_update),
        ("Device Handling", test_device_handling),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\nFAIL: Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for test_name, passed_test in results:
        status = "PASS" if passed_test else "FAIL"
        print(f"{status}: {test_name}")
    
    print("="*60)
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! Ready to train PPO.")
        print("\nNext steps:")
        print("1. Run: python train.py")
        print("2. Check output files (ppo_training_results.png, training_results.json)")
        print("3. Run: python analysis.py (for sensitivity analysis)")
    else:
        print("\nSome tests failed. Check errors above.")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
