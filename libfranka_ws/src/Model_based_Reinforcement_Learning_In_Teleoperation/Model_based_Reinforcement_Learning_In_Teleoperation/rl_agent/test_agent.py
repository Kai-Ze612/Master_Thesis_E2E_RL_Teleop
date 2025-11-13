"""
Test script for evaluating a trained Model-Based SAC agent.

This script loads a trained policy and evaluates it on the teleoperation task.
It supports multiple delay configurations and trajectory types for comprehensive testing.
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
from typing import Dict, Optional
from datetime import datetime

# Custom imports
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.training_env import TeleoperationEnvWithDelay
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.local_robot_simulator import TrajectoryType
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.sac_policy_network import StateEstimator, Actor
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    RNN_SEQUENCE_LENGTH,
    CHECKPOINT_DIR_RL,
    CHECKPOINT_DIR_LSTM,
)


def setup_logging() -> logging.Logger:
    """Configure logging for test script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def find_latest_checkpoint(checkpoint_dir: str, checkpoint_type: str = "best") -> Optional[str]:
    """Find the latest checkpoint of a given type."""
    import glob

    if checkpoint_type == "best":
        pattern = os.path.join(checkpoint_dir, "*/best_policy.pth")
    elif checkpoint_type == "final":
        pattern = os.path.join(checkpoint_dir, "*/final_policy.pth")
    else:
        pattern = os.path.join(checkpoint_dir, f"*/{checkpoint_type}")

    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None

    # Return most recent
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def load_model(checkpoint_path: str, device: torch.device, logger: logging.Logger) -> tuple:
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load state estimator (LSTM)
    state_estimator = StateEstimator().to(device)
    state_estimator.load_state_dict(checkpoint['state_estimator_state_dict'])
    state_estimator.eval()
    for param in state_estimator.parameters():
        param.requires_grad = False

    # Load actor
    actor = Actor().to(device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    logger.info("Model loaded successfully")
    logger.info(f"  Total timesteps: {checkpoint.get('total_timesteps', 'N/A')}")
    logger.info(f"  Total updates: {checkpoint.get('num_updates', 'N/A')}")

    return state_estimator, actor


def test_policy(
    state_estimator: StateEstimator,
    actor: Actor,
    env: TeleoperationEnvWithDelay,
    num_episodes: int,
    device: torch.device,
    deterministic: bool,
    render: bool,
    logger: logging.Logger
) -> Dict[str, float]:
    """Test the policy on the given environment."""

    episode_rewards = []
    episode_lengths = []
    episode_tracking_errors = []
    episode_prediction_errors = []
    episode_success = []  # Track successful completions

    logger.info(f"Testing policy for {num_episodes} episodes...")
    logger.info(f"  Deterministic actions: {deterministic}")
    logger.info(f"  Render: {render}")

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        tracking_errors = []
        prediction_errors = []

        # Reset hidden state for new episode
        hidden_state = state_estimator.init_hidden_state(1, device)

        done = False
        terminated = False

        while not done:
            # Get delayed sequence and remote state
            delayed_seq = env.get_delayed_target_buffer(RNN_SEQUENCE_LENGTH)
            delayed_seq = delayed_seq.reshape(1, RNN_SEQUENCE_LENGTH, N_JOINTS * 2)
            remote_state = env.get_remote_state().reshape(1, -1)

            # Select action
            with torch.no_grad():
                delayed_seq_t = torch.tensor(delayed_seq, dtype=torch.float32, device=device)
                remote_state_t = torch.tensor(remote_state, dtype=torch.float32, device=device)

                predicted_state_t, hidden_state = state_estimator(delayed_seq_t, hidden_state)

                actor_input_t = torch.cat([predicted_state_t, remote_state_t], dim=1)
                action_t, _, _ = actor.sample(actor_input_t, deterministic)
                action = action_t.cpu().numpy()[0]

            # Set predicted target before step
            env.set_predicted_target(predicted_state_t.cpu().numpy()[0])

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Track errors
            if 'real_time_joint_error' in info:
                tracking_errors.append(info['real_time_joint_error'])
            if 'prediction_error' in info and not np.isnan(info['prediction_error']):
                prediction_errors.append(info['prediction_error'])

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_success.append(1 if not terminated else 0)  # Success if not terminated early

        if tracking_errors:
            episode_tracking_errors.append(np.mean(tracking_errors))
        if prediction_errors:
            episode_prediction_errors.append(np.mean(prediction_errors))

        # Log episode summary
        logger.info(f"Episode {ep+1}/{num_episodes}: "
                   f"Reward={episode_reward:.2f}, "
                   f"Length={episode_length}, "
                   f"Success={'✓' if not terminated else '✗'}")

    # Calculate statistics
    test_metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.mean(episode_success) * 100,
        'mean_tracking_error': np.mean(episode_tracking_errors) if episode_tracking_errors else 0.0,
        'std_tracking_error': np.std(episode_tracking_errors) if episode_tracking_errors else 0.0,
        'mean_prediction_error': np.mean(episode_prediction_errors) if episode_prediction_errors else 0.0,
    }

    return test_metrics


def print_test_results(metrics: Dict[str, float], config: ExperimentConfig, trajectory: TrajectoryType, logger: logging.Logger):
    """Print formatted test results."""
    logger.info("\n" + "="*80)
    logger.info("TEST RESULTS")
    logger.info("="*80)
    logger.info(f"Configuration: {config.name}")
    logger.info(f"Trajectory: {trajectory.value}")
    logger.info("-"*80)
    logger.info(f"Mean Reward:        {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}")
    logger.info(f"Reward Range:       [{metrics['min_reward']:.3f}, {metrics['max_reward']:.3f}]")
    logger.info(f"Mean Episode Length: {metrics['mean_length']:.1f} steps")
    logger.info(f"Success Rate:       {metrics['success_rate']:.1f}%")
    logger.info("-"*80)
    logger.info(f"Mean Tracking Error:  {metrics['mean_tracking_error']*1000:.2f} ± {metrics['std_tracking_error']*1000:.2f} mm")
    if metrics['mean_prediction_error'] > 0:
        logger.info(f"Mean Prediction Error: {metrics['mean_prediction_error']*1000:.2f} mm")
    logger.info("="*80 + "\n")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Test a trained Model-Based SAC agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint file. If not specified, uses latest best_policy.pth")
    parser.add_argument("--checkpoint-type", type=str, default="best", choices=['best', 'final', 'early_stopped'],
                       help="Type of checkpoint to load if --checkpoint not specified")
    parser.add_argument("--config", type=str, default="2", choices=['1', '2', '3', '4'],
                       help="Delay configuration preset (1=LOW, 2=MEDIUM, 3=HIGH, 4=EXTREME)")
    parser.add_argument("--trajectory-type", type=str, default="figure_8",
                       choices=[t.value for t in TrajectoryType],
                       help="Reference trajectory type")
    parser.add_argument("--num-episodes", type=int, default=20,
                       help="Number of test episodes")
    parser.add_argument("--deterministic", action="store_true",
                       help="Use deterministic actions (recommended for testing)")
    parser.add_argument("--render", action="store_true",
                       help="Render the environment during testing")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    # Setup
    logger = setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Parse config and trajectory
    config_options = list(ExperimentConfig)
    CONFIG_MAP = {
        '1': config_options[0],
        '2': config_options[1],
        '3': config_options[2],
        '4': config_options[3]
    }
    config = CONFIG_MAP[args.config]
    trajectory_type = next(t for t in TrajectoryType if t.value.lower() == args.trajectory_type.lower())

    # Find or use specified checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        logger.info(f"Searching for {args.checkpoint_type} checkpoint...")
        checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR_RL, args.checkpoint_type)
        if not checkpoint_path:
            logger.error(f"No {args.checkpoint_type} checkpoint found in {CHECKPOINT_DIR_RL}")
            sys.exit(1)

    # Load model
    try:
        state_estimator, actor = load_model(checkpoint_path, device, logger)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Create test environment
    logger.info("\nCreating test environment...")
    logger.info(f"  Config: {config.name}")
    logger.info(f"  Trajectory: {trajectory_type.value}")

    render_mode = "human" if args.render else None
    env = TeleoperationEnvWithDelay(
        delay_config=config,
        trajectory_type=trajectory_type,
        randomize_trajectory=False,  # Fixed for reproducible testing
        seed=args.seed,
        render_mode=render_mode
    )

    # Run test
    try:
        test_metrics = test_policy(
            state_estimator=state_estimator,
            actor=actor,
            env=env,
            num_episodes=args.num_episodes,
            device=device,
            deterministic=args.deterministic,
            render=args.render,
            logger=logger
        )

        # Print results
        print_test_results(test_metrics, config, trajectory_type, logger)

    except KeyboardInterrupt:
        logger.warning("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"\nTest failed with error: {e}", exc_info=True)
    finally:
        env.close()
        logger.info("Test completed")


if __name__ == "__main__":
    main()
