"""
This script serves as a mujoco simulator for remote robot (follower).

The RL is trained to provide a compensation torque on top of a base PD controller.
"""

import mujoco
import numpy as np

from Hierarchical_Reinforcement_Learning_for_Adaptive_Control_Under_Stochastic_Network_Delays.controllers.pd_controller import PDController
from Hierarchical_Reinforcement_Learning_for_Adaptive_Control_Under_Stochastic_Network_Delays.controllers.inverse_kinematics import InverseKinematicsSolver


class RemoteRobotSimulator:
    def __init__(self, model_path: str, control_freq: int,
                 default_kp: np.ndarray, default_kd: np.ndarray,
                 torque_limits: np.ndarray, joint_limits_lower: np.ndarray,
                 joint_limits_upper: np.ndarray):
        """
        Initializes the remote robot simulator.
        """

        # Initialize MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.n_joints = 7
        self.ee_body_name = 'panda_hand'
        
        # Time step for velocity calculations
        self.dt = 1.0 / control_freq

        # Simulation frequency and substeps
        sim_freq = int(1.0 / self.model.opt.timestep)
        if sim_freq % control_freq != 0:
            raise ValueError("Simulation frequency must be a multiple of control frequency.")
        self.n_substeps = sim_freq // control_freq

        # Initialize controllers
        self.torque_limits = torque_limits
        self.pd_controller = PDController(
            kp=default_kp,
            kd=default_kd,
            torque_limits=torque_limits,
            joint_limits_lower=joint_limits_lower,
            joint_limits_upper=joint_limits_upper
        )

        # Initialize IK solver
        self.ik_solver = InverseKinematicsSolver(self.model, joint_limits_lower, joint_limits_upper)
        
        # State variables
        self.last_q_target = None
        self.last_time = 0.0

        # TCP offset from flange to end-effector (in meters
        self.tcp_offset = np.array([0.0, 0.0, 0.1034])
        
        # State variables
        self.current_q_target = np.zeros(self.n_joints)
        self.last_q_target = np.zeros(self.n_joints)
        self.smoothed_q_dot_target = np.zeros(self.n_joints)
        
        # Parameters for smoothing and filtering
        self.max_q_step = 0.01  # Max joint space interpolation step per cycle
        self.velocity_filter_alpha = 0.1  # EMA filter alpha for velocity
        self.max_joint_velocity = 0.75 # rad/s
        
        
    def reset(self, initial_qpos: np.ndarray):
        """Resets the robot to an initial joint configuration."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:self.n_joints] = initial_qpos
        self.data.qvel[:self.n_joints] = np.zeros(self.n_joints)
        mujoco.mj_forward(self.model, self.data)
        
        self.current_q_target = initial_qpos.copy()
        self.last_q_target = initial_qpos.copy()
        self.smoothed_q_dot_target.fill(0.0)

    def step(self, target_pos: np.ndarray, normalized_action: np.ndarray, characteristic_torque: float, action_delay_steps: int):
        """
        Executes one control step with RL-based torque compensation.
        """
        current_qpos = self.data.qpos[:self.n_joints].copy()
        current_qvel = self.data.qvel[:self.n_joints].copy()

        q_goal, _ = self.ik_solver.solver(
            target_pos,
            current_qpos,
            self.ee_body_name)

        if q_goal is None:
            q_goal = self.last_q_target
        
        q_error = q_goal - self.current_q_target
        error_norm = np.linalg.norm(q_error)
        
        if error_norm > self.max_q_step:
            step = q_error / error_norm * self.max_q_step
            self.current_q_target += step
        elif error_norm > 1e-6:
            self.current_q_target = q_goal.copy()
            
        q_target = self.current_q_target
        
        noisy_q_dot_target = (q_target - self.last_q_target) / self.dt
        noisy_q_dot_target = np.clip(noisy_q_dot_target, -self.max_joint_velocity, self.max_joint_velocity)
        
        self.smoothed_q_dot_target = (1 - self.velocity_filter_alpha) * self.smoothed_q_dot_target + \
                                     self.velocity_filter_alpha * noisy_q_dot_target
        
        self.last_q_target = q_target.copy()
        
        action_delay_seconds = action_delay_steps * self.dt
        predicted_joint_positions = current_qpos + current_qvel * action_delay_seconds
        predicted_joint_velocities = current_qvel # Assume constant velocity over small delay interval
        
        # Standard PD Controller Torque
        pd_torques = self.pd_controller.compute_desired_acceleration(
            target_positions=q_target,
            target_velocities= self.smoothed_q_dot_target,
            current_positions=predicted_joint_positions,
            current_velocities=predicted_joint_velocities
        )
        
        # RL-based Compensation Torque (De-normalization)
        compensation_torque = normalized_action * characteristic_torque

        # Final Torque Command
        tau_command = pd_torques + compensation_torque
        
        clipped_tau = np.clip(tau_command, -self.torque_limits, self.torque_limits)
        self.data.ctrl[:self.n_joints] = clipped_tau
        
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
            
    def get_state(self) -> dict:
        """Returns the current state of the follower robot."""
        return {
            "joint_pos": self.data.qpos[:self.n_joints].copy(),
            "joint_vel": self.data.qvel[:self.n_joints].copy()
        }
        
    def get_ee_position(self) -> np.ndarray:
        """
        Return the current end-effector (TCP) position in world coordinates.
        """
        
        # Get the position of the flange from MuJoCo
        ee_id = self.model.body(self.ee_body_name).id
        flange_position = self.data.xpos[ee_id].copy()

        # Add the offset to get the true TCP position
        tcp_position = flange_position + self.tcp_offset
        return tcp_position