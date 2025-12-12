import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_reward_logs(log_dir="reward_logs"):
    """Analyze reward logs to identify training issues"""
    
    log_path = Path(log_dir)
    
    # Find the most recent episode log file
    episode_files = list(log_path.glob("episode_summary_*.json"))
    if not episode_files:
        print("No episode log files found!")
        return
    
    latest_file = max(episode_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        episodes = json.load(f)
    
    if not episodes:
        print("No episode data found!")
        return
    
    print(f"Analyzing {len(episodes)} episodes...")
    
    # Extract data
    episode_nums = [ep['episode'] for ep in episodes]
    total_rewards = [ep['total_reward'] for ep in episodes]
    tracking_rewards = [ep['avg_tracking_reward'] for ep in episodes]
    action_penalties = [ep['avg_action_penalty'] for ep in episodes]
    cartesian_errors = [ep['avg_cartesian_error'] for ep in episodes]
    tracking_ratios = [ep['tracking_reward_ratio'] for ep in episodes]
    penalty_ratios = [ep['action_penalty_ratio'] for ep in episodes]
    
    # Analysis
    print("\n=== REWARD COMPONENT ANALYSIS ===")
    
    # Check for plateau detection
    recent_rewards = total_rewards[-50:] if len(total_rewards) >= 50 else total_rewards
    reward_std = np.std(recent_rewards)
    reward_mean = np.mean(recent_rewards)
    
    print(f"Recent 50 episodes - Mean: {reward_mean:.3f}, Std: {reward_std:.3f}")
    
    if reward_std < 0.1 * abs(reward_mean):
        print("⚠️  PLATEAU DETECTED: Reward variance is very low - agent may have stopped learning")
    
    # Component analysis
    recent_tracking = np.mean(tracking_rewards[-50:])
    recent_penalties = np.mean(action_penalties[-50:])
    recent_errors = np.mean(cartesian_errors[-50:])
    
    print(f"\nRecent Averages:")
    print(f"  Tracking Reward: {recent_tracking:.4f}")
    print(f"  Action Penalty: {recent_penalties:.4f}")
    print(f"  Cartesian Error: {recent_errors:.4f}m")
    print(f"  Tracking/Total Ratio: {np.mean(tracking_ratios[-50:]):.3f}")
    print(f"  Penalty/Total Ratio: {np.mean(penalty_ratios[-50:]):.3f}")
    
    # Identify dominant component
    if abs(recent_penalties) < 0.1 * recent_tracking:
        print("🔍 ACTION PENALTY is negligible - may not be providing learning signal")
    elif abs(recent_penalties) > 0.5 * recent_tracking:
        print("🔍 ACTION PENALTY is dominating - may be hindering exploration")
    
    # Error analysis
    error_threshold_count = sum(1 for ep in episodes[-50:] if ep['avg_cartesian_error'] < 0.1)
    print(f"\nEpisodes with avg error < 0.1m: {error_threshold_count}/50")
    
    if error_threshold_count > 40:
        print("🔍 Most episodes have low error - reward function may be saturated")
    elif error_threshold_count < 5:
        print("🔍 High error episodes - agent struggling with basic tracking")
    
    # Learning progression analysis
    if len(episodes) >= 100:
        early_mean = np.mean(total_rewards[:50])
        late_mean = np.mean(total_rewards[-50:])
        improvement = late_mean - early_mean
        
        print(f"\nLearning Progression:")
        print(f"  Early episodes (1-50): {early_mean:.3f}")
        print(f"  Recent episodes: {late_mean:.3f}")
        print(f"  Improvement: {improvement:.3f}")
        
        if abs(improvement) < 0.05:
            print("⚠️  MINIMAL IMPROVEMENT: Agent may not be learning effectively")
    
    # Visualization
    create_reward_plots(episodes, log_dir)
    
    return {
        'plateau_detected': reward_std < 0.1 * abs(reward_mean),
        'recent_tracking_reward': recent_tracking,
        'recent_action_penalty': recent_penalties,
        'recent_error': recent_errors,
        'low_error_episodes': error_threshold_count,
        'action_penalty_negligible': abs(recent_penalties) < 0.1 * recent_tracking
    }

def create_reward_plots(episodes, log_dir="reward_logs"):
    """Create visualization plots for reward analysis"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Reward Component Analysis', fontsize=16)
    
    episode_nums = [ep['episode'] for ep in episodes]
    
    # Plot 1: Total Reward
    axes[0, 0].plot(episode_nums, [ep['total_reward'] for ep in episodes])
    axes[0, 0].set_title('Total Reward')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Plot 2: Reward Components
    axes[0, 1].plot(episode_nums, [ep['avg_tracking_reward'] for ep in episodes], label='Tracking', alpha=0.7)
    axes[0, 1].plot(episode_nums, [ep['avg_action_penalty'] for ep in episodes], label='Action Penalty', alpha=0.7)
    axes[0, 1].set_title('Reward Components')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Cartesian Error
    axes[0, 2].plot(episode_nums, [ep['avg_cartesian_error'] for ep in episodes])
    axes[0, 2].axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='0.1m threshold')
    axes[0, 2].set_title('Average Cartesian Error')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Error (m)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Plot 4: Reward Ratios
    axes[1, 0].plot(episode_nums, [ep['tracking_reward_ratio'] for ep in episodes], label='Tracking Ratio')
    axes[1, 0].plot(episode_nums, [ep['action_penalty_ratio'] for ep in episodes], label='Penalty Ratio')
    axes[1, 0].set_title('Component Ratios')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 5: Moving Average (smoothed trends)
    window = min(20, len(episodes)//5)
    if window > 1:
        smoothed_rewards = np.convolve([ep['total_reward'] for ep in episodes], 
                                     np.ones(window)/window, mode='valid')
        smoothed_episodes = episode_nums[window-1:]
        axes[1, 1].plot(smoothed_episodes, smoothed_rewards)
        axes[1, 1].set_title(f'Smoothed Total Reward (window={window})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].grid(True)
    
    # Plot 6: Error Distribution
    errors = [ep['avg_cartesian_error'] for ep in episodes]
    axes[1, 2].hist(errors, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 2].axvline(x=0.1, color='r', linestyle='--', label='0.1m threshold')
    axes[1, 2].set_title('Error Distribution')
    axes[1, 2].set_xlabel('Cartesian Error (m)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(log_dir) / 'reward_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {plot_path}")
    plt.show()

def quick_diagnosis(log_dir="reward_logs"):
    """Quick diagnosis of current training state"""
    results = analyze_reward_logs(log_dir)
    
    if results is None:
        return
        
    print("\n" + "="*50)
    print("QUICK DIAGNOSIS")
    print("="*50)
    
    if results['plateau_detected']:
        print("🚨 PLATEAU DETECTED - Agent has likely stopped learning")
        
        if results['action_penalty_negligible']:
            print("   → Action penalty is too small - increase weight")
            
        if results['low_error_episodes'] > 40:
            print("   → Tracking error consistently low - reward may be saturated")
            print("   → Consider tightening error tolerance or changing reward shape")
    
    if results['recent_error'] > 0.3:
        print("🚨 HIGH TRACKING ERROR - Agent struggling with basic task")
        print("   → Check PD gains, action scaling, or reward function")
    
    print("\nRECOMMENDED ACTIONS:")
    if results['action_penalty_negligible']:
        print("1. Increase action penalty weight from 0.01 to 0.05-0.1")
    if results['plateau_detected']:
        print("2. Modify reward function to be less saturated")
        print("3. Add exploration bonus or curriculum learning")
    if results['recent_error'] < 0.05:
        print("4. Task may be too easy - increase difficulty")

# Example usage
if __name__ == "__main__":
    # Run analysis
    quick_diagnosis()