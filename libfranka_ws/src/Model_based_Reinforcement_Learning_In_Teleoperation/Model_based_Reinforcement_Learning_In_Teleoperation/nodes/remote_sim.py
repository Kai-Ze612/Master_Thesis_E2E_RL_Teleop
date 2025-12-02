"""
Pure Simulation Remote Robot Node.
Acts as a "Digital Twin" replacing the real hardware.

Pipeline:
1. Receives Agent Commands (Predict Target + RL Torque).
2. Calculates Torque (PD + Inverse Dynamics + RL).
3. Steps MuJoCo Physics (instead of sending to hardware).
4. Publishes Joint States (so the Agent thinks it's talking to a real robot).
5. Renders visuals.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PointStamped

import mujoco
import mujoco.viewer
import threading
import numpy as np
import time
from collections import deque

from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import (
    N_JOINTS,
    DEFAULT_MUJOCO_MODEL_PATH,
    DEFAULT_CONTROL_FREQ,
    INITIAL_JOINT_CONFIG,
    TORQUE_LIMITS,
    DEFAULT_KD_REMOTE,
    DEFAULT_KP_REMOTE,
    EE_BODY_NAME,
    DEPLOYMENT_HISTORY_BUFFER_SIZE,
)

class SimRemoteRobotNode(Node):
    def __init__(self):
        super().__init__('sim_remote_robot_node')
        
        # --- Config ---
        self.n_joints = N_JOINTS
        self.freq = DEFAULT_CONTROL_FREQ
        self.dt = 1.0 / self.freq
        self.joint_names = [f'panda_joint{i+1}' for i in range(self.n_joints)]
        
        # --- MuJoCo Setup (The Physics Engine) ---
        self.model = mujoco.MjModel.from_xml_path(DEFAULT_MUJOCO_MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.ee_body_id = self.model.body(EE_BODY_NAME).id
        
        # Set Initial State
        self.data.qpos[:self.n_joints] = INITIAL_JOINT_CONFIG
        self.data.qvel[:self.n_joints] = 0.0
        # Warmup physics to settle
        for _ in range(50): 
            mujoco.mj_step(self.model, self.data)

        # --- Control State ---
        self.target_q = INITIAL_JOINT_CONFIG.copy()
        self.target_qd = np.zeros(self.n_joints)
        self.tau_rl = np.zeros(self.n_joints)
        
        # --- Delays ---
        self.declare_parameter('experiment_config', ExperimentConfig.LOW_DELAY.value)
        self.declare_parameter('seed', 50)
        
        config_int = self.get_parameter('experiment_config').value
        seed = self.get_parameter('seed').value
        self.delay_sim = DelaySimulator(self.freq, ExperimentConfig(config_int), seed)
        
        self.torque_queue = deque(maxlen=DEPLOYMENT_HISTORY_BUFFER_SIZE)

        # --- ROS2 Interfaces ---
        # 1. Subscribe to Agent
        self.create_subscription(JointState, 'agent/predict_target', self.target_cb, 10)
        self.create_subscription(Float64MultiArray, 'agent/tau_rl', self.tau_rl_cb, 10)
        
        # 2. Publish "Fake" Robot State (Agent needs this)
        self.state_pub = self.create_publisher(JointState, '/franka/joint_states', 10) # Mimic hardware topic
        self.ee_pub = self.create_publisher(PointStamped, 'remote_robot/ee_pose', 10)

        # 3. Main Physics Loop (High Frequency)
        self.timer = self.create_timer(self.dt, self.physics_loop)
        
        # 4. Viewer Thread
        self.viewer_thread = threading.Thread(target=self._run_viewer, daemon=True)
        self.viewer_thread.start()
        
        self.get_logger().info("SIMULATION STARTED. MuJoCo Physics Active.")

    def _run_viewer(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.azimuth = 135
            viewer.cam.distance = 2.0
            viewer.cam.lookat[:] = [0.5, 0.0, 0.5]
            while viewer.is_running() and rclpy.ok():
                viewer.sync()
                time.sleep(0.03)

    def target_cb(self, msg):
        self.target_q = np.array(msg.position[:self.n_joints])
        self.target_qd = np.array(msg.velocity[:self.n_joints])

    def tau_rl_cb(self, msg):
        self.tau_rl = np.array(msg.data)

    def _get_required_torque(self, q, v, a_des):
        """
        Calculate total torque needed to achieve acceleration 'a_des'.
        Output = M(q)*a_des + C(q,v)*v + G(q)
        """
        # Save state
        q_old = self.data.qpos.copy()
        v_old = self.data.qvel.copy()
        acc_old = self.data.qacc.copy()
        
        # Set desired state
        self.data.qpos[:self.n_joints] = q
        self.data.qvel[:self.n_joints] = v
        self.data.qacc[:self.n_joints] = a_des
        
        # Inverse Dynamics
        mujoco.mj_inverse(self.model, self.data)
        tau_required = self.data.qfrc_inverse[:self.n_joints].copy()
        
        # Restore state
        self.data.qpos[:] = q_old
        self.data.qvel[:] = v_old
        self.data.qacc[:] = acc_old
        
        return tau_required

    def physics_loop(self):
        # 1. Current State
        q = self.data.qpos[:self.n_joints].copy()
        qd = self.data.qvel[:self.n_joints].copy()
        
        # 2. PD Control (Calculate Desired Acceleration)
        # We want to accelerate towards the target
        q_err = (self.target_q - q) 
        qd_err = (self.target_qd - qd)
        acc_des = DEFAULT_KP_REMOTE * q_err + DEFAULT_KD_REMOTE * qd_err
        
        # 3. Inverse Dynamics
        # Calculate the torque needed to achieve 'acc_des' perfectly
        # This includes Gravity, Coriolis, and Inertia.
        tau_id = self._get_required_torque(q, qd, acc_des)
        
        # 4. Add RL Compensation
        # The RL agent learns to output a 'residual' torque on top of the ideal controller
        tau_total = tau_id + self.tau_rl
        
        # 5. Clip
        tau_total = np.clip(tau_total, -TORQUE_LIMITS, TORQUE_LIMITS)
        
        # 6. Action Delay Simulation
        self.torque_queue.append(tau_total)
        delay_steps = self.delay_sim.get_action_delay_steps()
        
        if delay_steps >= len(self.torque_queue):
            # If buffer not full, apply gravity comp only (hold position)
            # (Simplification: just zero is unsafe, better to use gravity comp)
            # Re-calculate gravity for current pose
            tau_applied = self._get_required_torque(q, np.zeros_like(q), np.zeros_like(q))
        else:
            tau_applied = self.torque_queue[-1 - delay_steps]
            
        # 7. Apply to Physics
        tau_applied[-1] = 0.0 # Safety for gripper
        self.data.ctrl[:self.n_joints] = tau_applied
        
        # Substeps for stability (optional, but good practice)
        mujoco.mj_step(self.model, self.data)
        
        # 8. Publish
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = q.tolist()
        msg.velocity = qd.tolist()
        self.state_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SimRemoteRobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()