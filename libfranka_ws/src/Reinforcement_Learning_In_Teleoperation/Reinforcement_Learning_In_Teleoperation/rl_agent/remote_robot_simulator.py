"""
This script serves as a mujoco simulator for remote robot (follower).
"""

import mujoco
import numpy as np

from Reinforcement_Learning_In_Teleoperation.controllers.inverse_kinematics import InverseKinematicsSolver

class RemoteRobotSimulator:
    def __init__(self, 
                 model_path: str, 
                 control_freq: int,
                 torque_limits: np.ndarray, 
                 joint_limits_lower: np.ndarray,
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
        self.joint_limits_lower = joint_limits_lower
        self.joint_limits_upper = joint_limits_upper

        # Initialize IK solver
        self.ik_solver = InverseKinematicsSolver(self.model, joint_limits_lower, joint_limits_upper)
    
        # TCP offset from flange to end-effector (in meters
        self.tcp_offset = np.array([0.0, 0.0, 0.1034])
        
        # State variables
        self.current_q_target = np.zeros(self.n_joints)
        self.last_q_target = np.zeros(self.n_joints)
        self.last_time = 0.0

    def reset(self, initial_qpos: np.ndarray):
        """Resets the robot to an initial joint configuration."""
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial joint positions and velocities
        self.data.qpos[:self.n_joints] = initial_qpos
        self.data.qvel[:self.n_joints] = np.zeros(self.n_joints)
        
        # Send the simulation forward to update derived quantities
        mujoco.mj_forward(self.model, self.data)
        
        self.current_q_target = initial_qpos.copy()
        self.last_q_target = initial_qpos.copy()

    def step(self, target_pos: np.ndarray, normalized_action: np.ndarray, characteristic_torque: float, action_delay_steps: int):
        """
        Executes one control step with RL-based torque compensation.
        """
        # Get current robot state
        current_qpos = self.data.qpos[:self.n_joints].copy()
        
        # Convert target position to joint space using IK       
        q_target, _ = self.ik_solver.solver(
            target_pos,
            current_qpos,
            self.ee_body_name)

        # Handle IK failure
        if q_target is None:
            q_target = self.last_q_target
        else:
            self.current_q_target = q_target.copy()
            
        self.last_q_target = q_target.copy()
        
        # Final Torque Command
        tau_command = normalized_action * characteristic_torque
        
        clipped_tau = np.clip(tau_command, -self.torque_limits, self.torque_limits)
        self.data.ctrl[:self.n_joints] = clipped_tau
        
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
            
    def get_state(self) -> dict:
        """
        Returns the current state of the follower robot.
        This will be used by RL observation space.
        """
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

    def get_current_q_target(self) -> np.ndarray:
        """
        Get current target joint positions only for debugging purposes.
        """
        return {
            "q_target": self.current_q_target.copy(),
            "last_q_target": self.last_q_target.copy()
        }