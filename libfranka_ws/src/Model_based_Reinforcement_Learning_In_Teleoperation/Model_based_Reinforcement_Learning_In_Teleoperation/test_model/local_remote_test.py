"""
direct_teleop_test.py

A Pure Classical Control Node for System Identification.
Path: Local Robot -> [PD Controller + Gravity Comp] -> Remote Robot
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import mujoco
import numpy as np
import sys

# --- CONFIG IMPORTS ---
# We only need physical constants, no AI configs
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    DEFAULT_MUJOCO_MODEL_PATH, 
    DEFAULT_CONTROL_FREQ,
    INITIAL_JOINT_CONFIG,
    TORQUE_LIMITS,
    DEFAULT_KD_REMOTE,
    DEFAULT_KP_REMOTE,
    EE_BODY_NAME,
)

class DirectTeleopTest(Node):
    def __init__(self):
        super().__init__('direct_teleop_test_node')
        
        self.n_joints = N_JOINTS
        self.joint_names = [f'panda_joint{i+1}' for i in range(self.n_joints)]
        
        # 1. Control Parameters (Classic PD)
        self.kp = DEFAULT_KP_REMOTE
        self.kd = DEFAULT_KD_REMOTE
        self.torque_limits = TORQUE_LIMITS
        
        # 2. State Variables
        self.q_remote = INITIAL_JOINT_CONFIG.copy()
        self.qd_remote = np.zeros(self.n_joints)
        self.q_leader = INITIAL_JOINT_CONFIG.copy()
        self.qd_leader = np.zeros(self.n_joints)
        
        # 3. Mujoco Setup (Solely for Gravity Compensation)
        try:
            self.mj_model = mujoco.MjModel.from_xml_path(DEFAULT_MUJOCO_MODEL_PATH)
            self.mj_data = mujoco.MjData(self.mj_model)
        except Exception as e:
            self.get_logger().fatal(f"Could not load Mujoco model for G-Comp: {e}")
            sys.exit(1)

        # 4. Connectivity Flags
        self.remote_ready = False
        self.leader_ready = False
        self.safety_lock_disengaged = False

        # 5. ROS2 Interfaces
        # Listen to the LEADER (The Target)
        self.sub_leader = self.create_subscription(
            JointState, 
            'local_robot/joint_states', 
            self.leader_callback, 
            10
        )
        
        # Listen to the FOLLOWER (The Real Robot)
        self.sub_remote = self.create_subscription(
            JointState, 
            'remote_robot/joint_states', 
            self.remote_callback, 
            10
        )
        
        # Command the FOLLOWER
        self.pub_torque = self.create_publisher(
            Float64MultiArray, 
            'joint_tau/torques_desired', 
            10
        )
        
        # Control Loop Timer (1kHz)
        self.dt = 1.0 / DEFAULT_CONTROL_FREQ
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        self.get_logger().info("=== DIRECT TELEOP (NO-AI) TEST STARTED ===")
        self.get_logger().info("Waiting for robot connections...")

    def leader_callback(self, msg):
        try:
            # Parse Leader State
            name_map = {name: i for i, name in enumerate(msg.name)}
            self.q_leader = np.array([msg.position[name_map[n]] for n in self.joint_names])
            self.qd_leader = np.array([msg.velocity[name_map[n]] for n in self.joint_names])
            
            if not self.leader_ready:
                self.get_logger().info("Leader Robot Signal Detected.")
                self.leader_ready = True
        except Exception as e:
            pass

    def remote_callback(self, msg):
        try:
            # Parse Remote State
            name_map = {name: i for i, name in enumerate(msg.name)}
            self.q_remote = np.array([msg.position[name_map[n]] for n in self.joint_names])
            self.qd_remote = np.array([msg.velocity[name_map[n]] for n in self.joint_names])
            
            if not self.remote_ready:
                self.get_logger().info("Remote Robot Signal Detected.")
                self.remote_ready = True
        except Exception as e:
            pass

    # def _get_gravity_comp(self, q):
    #     """
    #     Calculate G(q) using Mujoco Inverse Dynamics.
    #     Set vel/acc to 0 to isolate gravity term.
    #     """
    #     self.mj_data.qpos[:self.n_joints] = q
    #     self.mj_data.qvel[:self.n_joints] = 0.0
    #     self.mj_data.qacc[:self.n_joints] = 0.0
    #     mujoco.mj_inverse(self.mj_model, self.mj_data)
    #     return self.mj_data.qfrc_inverse[:self.n_joints].copy()

    def _normalize_angle(self, angle):
        """Normalize to [-pi, pi] to handle joint wrapping."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def control_loop(self):
        # 1. Safety: Wait for data
        if not self.remote_ready or not self.leader_ready:
            return

        # 2. Safety: Prevent "The Jump"
        # If the leader and remote are too far apart at startup, do NOT engage PD.
        # Just output Gravity Compensation to hold position.
        if not self.safety_lock_disengaged:
            error_norm = np.linalg.norm(self.q_leader - self.q_remote)
            if error_norm > 0.5: # 0.3 rad threshold
                self.get_logger().warn(
                    f"SAFETY LOCK ACTIVE: Alignment Error = {error_norm:.2f} rad. "
                    f"Sync Leader/Remote positions manually.", 
                    throttle_duration_sec=1.0
                )
                # Hold current position against gravity
                tau_g = self._get_gravity_comp(self.q_remote)
                self._publish_torque(tau_g)
                return
            else:
                self.get_logger().info("Safety Lock Disengaged. PD Control ENGAGED.")
                self.safety_lock_disengaged = True

        # --- CONTROL LAW ---
        
        # A. Gravity Compensation (G)
        # tau_g = self._get_gravity_comp(self.q_remote)
        
        # B. PD Controller
        # Position Error
        q_err = self._normalize_angle(self.q_leader - self.q_remote)
        # Velocity Error (Feedforward - Feedback)
        qd_err = self.qd_leader - self.qd_remote 
        
        tau_pd = (self.kp * q_err) + (self.kd * qd_err)
        
        # C. Total Torque
        tau_total = tau_pd
        
        # D. Clip and Safety
        tau_total = np.clip(tau_total, -self.torque_limits, self.torque_limits)
        tau_total[-1] = 0.0  # Gripper is usually position controlled or ignored
        
        # E. Publish
        self._publish_torque(tau_total)

    def _publish_torque(self, tau):
        msg = Float64MultiArray()
        msg.data = tau.tolist()
        self.pub_torque.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DirectTeleopTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()