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
Master_Study_Master_Thesis/
├── src/
│   ├── e2e_rl/              # [Proposed Method] LSTM-based End-to-End Policy
│   ├── baselines/           # Comparative Models
│   │   ├── state_aug/       # Baseline 1: State Augmentation (Frame Stacking)
│   │   └── model_based/     # Baseline 2: Model-Based Dynamics Approach
│   ├── envs/                # Teleoperation Simulation (Franka Panda / Gymnasium)
│   └── utils/               # Signal processing & Delay emulation
├── configs/                 # Hyperparameters (PPO/SAC, LSTM, etc.)
├── results/                 # Comparative plots and metrics
├── data/                    # Stochastic delay profiles
└── Thesis_Paper.pdf         # Full research paper
