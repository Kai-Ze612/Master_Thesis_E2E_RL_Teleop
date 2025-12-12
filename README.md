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

```text
libfranka_ws/
├── src/
│   ├── E2E_Teleoperation/             # [Proposed Method] Novel End-to-End LSTM-based Policy
│   │                                  # (The core contribution of this Master's Thesis)
│   │
│   ├── Model_Based_RL_Teleoperation/  # [Previous Iteration] Dynamics-aware RL framework
│   │                                  # (Submitted to IFAC 2026)
│   │
│   ├── Hierarchical_RL_Teleoperation/ # [Previous Iteration] Multi-level architecture
│   │                                  # (Submitted to ICRA 2026)
│   │
│   ├── SBSP/                          # [Baseline] SOTA Model-Based RL Framework
│   │                                  # (Benchmarking candidate for sample efficiency)
│   │
│   └── A-SAC/                         # [Baseline] SOTA Model-Free RL Framework
│   │                                  # (Benchmarking candidate for asymptotic performance)
│   │
│   ├── mujoco_ros_pkgs/               # [Simulation] MuJoCo physics engine interface for ROS 2
│   │
│   └── multipanda_ros2/               # [Simulation] Franka Panda robot descriptions & scenes
│                                      # (Adapted from: [github.com/tenfoldpaper/multipanda_ros2]

