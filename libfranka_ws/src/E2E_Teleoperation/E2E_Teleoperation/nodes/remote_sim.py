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

from E2E_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig

import E2E_Teleoperation.config.robot_config as cfg


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

    def _get_inverse_dynamics(self, q, v, a_des):
        """ Calculate Inertial+Coriolis torque (No Gravity). """
        # We use a COPY of data for calculation to not mess up physics integration
        # Note: In pure sim, we can use the main data structure CAREFULLY, 
        # or better, use `mujoco.mj_rne` for specific terms.
        # Here we use the standard inverse approach:
        
        # Save current state
        q_old = self.data.qpos.copy()
        v_old = self.data.qvel.copy()
        acc_old = self.data.qacc.copy()
        
        self.data.qpos[:self.n_joints] = q
        self.data.qvel[:self.n_joints] = v
        self.data.qacc[:self.n_joints] = a_des
        
        mujoco.mj_inverse(self.model, self.data)
        tau_full = self.data.qfrc_inverse[:self.n_joints].copy()
        
        # Calculate Gravity Only
        self.data.qacc[:self.n_joints] = 0.0
        self.data.qvel[:self.n_joints] = 0.0
        mujoco.mj_inverse(self.model, self.data)
        tau_g = self.data.qfrc_inverse[:self.n_joints].copy()
        
        # Restore State
        self.data.qpos[:] = q_old
        self.data.qvel[:] = v_old
        self.data.qacc[:] = acc_old
        
        # In Simulation, MuJoCo *DOES* apply gravity naturally.
        # So our controller needs to CANCEL it (Gravity Compensation).
        # Controller Output = (M*a + C*v) + G
        #
        # WAIT! In the REAL ROBOT script, we removed G because Franka adds it.
        # In MUJOCO SIMULATION, we MUST Provide G if we want to hold the arm.
        # UNLESS we set gravity=0 in xml, or we implement gravity comp manually.
        
        # Let's emulate the Franka Hardware Controller:
        # The "Simulated Motor" receives `tau_command`.
        # Real Franka adds `tau_g` internally. 
        # So we should ADD `tau_g` to the final output to simulate Franka firmware behavior.
        
        return (tau_full - tau_g), tau_g 

    def physics_loop(self):
        # 1. Current State
        q = self.data.qpos[:self.n_joints].copy()
        qd = self.data.qvel[:self.n_joints].copy()
        
        # 2. PD Control
        q_err = (self.target_q - q) # Simple diff, normalize if needed
        qd_err = (self.target_qd - qd)
        acc_des = DEFAULT_KP_REMOTE * q_err + DEFAULT_KD_REMOTE * qd_err
        
        # 3. Compute Torque
        # tau_inertial = M*a + C*v
        tau_inertial, tau_gravity = self._get_inverse_dynamics(q, qd, acc_des)
        
        # 4. Combine (Emulate Franka Logic)
        # Agent sends: tau_inertial + tau_rl
        tau_command_user = tau_inertial + self.tau_rl
        tau_command_user = np.clip(tau_command_user, -TORQUE_LIMITS, TORQUE_LIMITS)
        
        # Emulate Action Delay
        self.torque_queue.append(tau_command_user)
        delay = self.delay_sim.get_action_delay_steps()
        if delay >= len(self.torque_queue):
            applied_user_torque = np.zeros(self.n_joints)
        else:
            applied_user_torque = self.torque_queue[-1 - delay]
            
        # 5. Physics Integration
        # Total Torque applied to physics = User_Command + Gravity_Comp(Firmware)
        tau_total = applied_user_torque + tau_gravity
        
        # Safety
        tau_total[-1] = 0.0 # Gripper
        
        self.data.ctrl[:self.n_joints] = tau_total
        mujoco.mj_step(self.model, self.data)
        
        # 6. Publish State (Feedback to Agent)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = q.tolist()
        msg.velocity = qd.tolist()
        self.state_pub.publish(msg)
        
        # 7. Debug
        ee_pos = self.data.body(EE_BODY_NAME).xpos
        pt = PointStamped()
        pt.header = msg.header
        pt.header.frame_id = "world"
        pt.point.x, pt.point.y, pt.point.z = ee_pos
        self.ee_pub.publish(pt)

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