# End-to-End Reinforcement Learning for Robust Teleoperation under Stochastic Delays

This repository contains the implementation, simulation environment, and experimental results of the Master's Thesis **"End-to-End Reinforcement Learning for Robust Teleoperation under Stochastic Delays"** conducted at the **Technical University of Munich (TUM)**.

## Project Overview

Stochastic communication delays in teleoperation introduce signal discontinuities that undermine control stability and degrade performance.

To address this, this research proposes a **End-to-End Deep Reinforcement Learning (RL)** framework. By incorporating **Long Short-Term Memory (LSTM)** units directly into the policy network, the agent effectively handles the partial observability caused by delays. It learns to reconstruct the system state internally and directly outputs **optimal torque commands**, balancing tracking accuracy with velocity smoothness without relying on separate state estimators or classical controllers.

## Author
**Kai-Ze Deng** M.Sc. Robotics, Cognition and Intelligence  
Department of Informatics  
Technical University of Munich (TUM)

## Supervisor
**Zewen Yang**
Postdoctoral Researcher  
Munich Institute of Robotics and Machine Intelligence (MIRMI)  
Technical University of Munich (TUM)  

## Repository Structure

The repository tracks the progressive development of the solution, containing the baselines and the final proposed framework.

## Repository Structure

The repository is organized as a ROS 2 workspace (`libfranka_ws`). Each major framework is contained within its own package, following a modular structure (Config, Nodes, Utils, and Core Algorithms).

```text
libfranka_ws/
├── src/
│   ├── E2E_Teleoperation/                 # [Proposed Method] Novel End-to-End LSTM-based Policy
│   │   ├── config/                        # Hyperparameter configurations
│   │   ├── E2E_RL/                        # Core Reinforcement Learning implementation
│   │   │   ├── sac_policy_network.py      # Network architecture (Actor-Critic + LSTM)
│   │   │   ├── sac_training_algorithm.py  # SAC algorithm logic
│   │   │   ├── training_env.py            # Gymnasium environment wrapper
│   │   │   ├── train_agent.py             # Main training entry point
│   │   │   ├── local_robot_simulator.py   # Leader robot physics/simulation
│   │   │   └── remote_robot_simulator.py  # Follower robot physics/simulation
│   │   ├── nodes/                         # ROS 2 Nodes for deployment
│   │   └── utils/                         # Shared utilities (delay simulator, IK solver)
│   │
│   ├── Model_Based_RL_Teleoperation/      # [Previous Iteration] Dynamics-aware RL framework
│   │                                      # (Contains: Dynamics Models, MPC Controllers, Data Buffers)
│   │
│   ├── Hierarchical_RL_Teleoperation/     # [Previous Iteration] Multi-level control architecture
│   │                                      # (Contains: High-level Planner, Low-level Controller)
│   │
│   ├── SBSP/                              # [Baseline] SOTA Model-Based RL Framework
│   │
│   ├── A-SAC/                             # [Baseline] SOTA Model-Free RL Framework
│   │
│   ├── mujoco_ros_pkgs/                   # [Simulation] MuJoCo physics engine interface for ROS2
│   │
│   └── multipanda_ros2/                   # [Simulation] Franka Panda robot descriptions & scenes
│   |                                      # (Adapted from: [github.com/tenfoldpaper/multipanda_ros2]

