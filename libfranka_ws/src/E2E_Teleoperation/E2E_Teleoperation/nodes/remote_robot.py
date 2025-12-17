"""
The script is the remote robot simulator (Forward Dynamics).

Pipeline:
1. Subscribe to 'agent/tau_rl' (The calculated torque from the RL agent)
2. Subscribe to 'local_robot/joint_states' (The Ground Truth / Leader State)
3. Apply torque to MuJoCo physics engine (qfrc_applied)
4. Step MuJoCo simulation (Forward Dynamics)
5. Publish 'remote_robot/joint_states' (The resulting state of this robot)
6. Calculate and log Real-Time Tracking Error (Remote vs Local)
"""


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PointStamped
import mujoco
import mujoco.viewer  # Import viewer for real-time visualization
import numpy as np
from collections import deque

# Custom imports
from E2E_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
import E2E_Teleoperation.config.robot_config as cfg

class RemoteRobotNode(Node):
    def __init__(self):
        super().__init__('remote_robot_node')
        
        # --- 1. Configurations ---
        self.n_joints = cfg.N_JOINTS
        self.control_freq = cfg.CONTROL_FREQ
        self.dt = 1.0 / self.control_freq
        self.joint_names = [f'panda_joint{i+1}' for i in range(self.n_joints)]
        
        # --- 2. MuJoCo Setup (Forward Dynamics) ---
        self.model_path = str(cfg.DEFAULT_MUJOCO_MODEL_PATH)
        self.mj_model = mujoco.MjModel.from_xml_path(self.model_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # Set Initial State (Warm Start)
        self.mj_data.qpos[:self.n_joints] = cfg.INITIAL_JOINT_CONFIG
        self.mj_data.qvel[:self.n_joints] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Launch the Passive Viewer (Non-blocking GUI)
        self.viewer = mujoco.viewer.launch_passive(
            self.mj_model, 
            self.mj_data, 
            key_callback=lambda key: self._on_key(key)
        )

        # --- 3. Delay & Experiment Config ---
        self.declare_parameter('experiment_config', ExperimentConfig.HIGH_VARIANCE.value)
        self.declare_parameter('seed', 42)
        
        exp_config_val = self.get_parameter('experiment_config').value
        seed_val = self.get_parameter('seed').value
        self.delay_config = ExperimentConfig(exp_config_val)
        
        self.action_delay_sim = DelaySimulator(self.control_freq, self.delay_config, seed=seed_val)
        self.torque_history = deque(maxlen=cfg.BUFFER_SIZE)

        # --- 4. State Variables ---
        self.current_rl_tau = np.zeros(self.n_joints)
        self.tau_received = False
        
        # Ground Truth (Leader) State for Error Calculation
        self.local_q = cfg.INITIAL_JOINT_CONFIG.copy()
        self.local_state_received = False

        # --- 5. ROS2 Subscribers ---
        # A. Subscribe ONLY to the Agent's Torque Output
        self.tau_sub = self.create_subscription(
            Float64MultiArray, 
            'agent/tau_rl', 
            self.tau_callback, 
            10
        )

        # B. Subscribe to Local Robot (Leader) for Error Tracking ONLY
        self.local_state_sub = self.create_subscription(
            JointState,
            'local_robot/joint_states',
            self.local_state_callback,
            10
        )

        # --- 6. ROS2 Publishers ---
        self.joint_state_pub = self.create_publisher(JointState, 'remote_robot/joint_states', 10)
        self.ee_pose_pub = self.create_publisher(PointStamped, 'remote_robot/ee_pose', 10)

        # --- 7. Main Simulation Loop ---
        self.timer = self.create_timer(self.dt, self.simulation_step)
        
        self.get_logger().info(f"Remote Robot Simulator Started.")

    def _on_key(self, key):
        """Handle viewer key presses (optional)."""
        pass

    def tau_callback(self, msg: Float64MultiArray):
        """Receive TOTAL torque from RL Agent."""
        self.current_rl_tau = np.array(msg.data, dtype=np.float32)
        if not self.tau_received:
            self.tau_received = True
            self.get_logger().info("First Torque command received.")

    def local_state_callback(self, msg: JointState):
        """Update the Ground Truth (Leader) state for error checking."""
        try:
            name_map = {name: i for i, name in enumerate(msg.name)}
            self.local_q = np.array([msg.position[name_map[n]] for n in self.joint_names])
            self.local_state_received = True
        except Exception:
            pass

    def simulation_step(self):
        """
        1. Apply Delayed Torque (Sole Control Input)
        2. Step MuJoCo Physics (Forward Dynamics)
        3. Sync Viewer
        4. Publish State
        5. LOGGING (Desired Q, True Q, RL Tau)
        """
        # 1. Apply Delay Simulation
        self.torque_history.append(self.current_rl_tau)
        delay_steps = self.action_delay_sim.get_action_delay_steps()
        
        if len(self.torque_history) > delay_steps:
            applied_tau = self.torque_history[-1 - delay_steps]
        else:
            # If buffer not full or just starting, apply zero torque
            applied_tau = np.zeros(self.n_joints)

        # 2. Physics Step (Forward Dynamics)
        # CRITICAL: We overwrite qfrc_applied with the Agent's torque.
        # This is the "Total Control" - no other forces (except internal physics like gravity/friction) are added manually.
        self.mj_data.qfrc_applied[:self.n_joints] = applied_tau
        
        mujoco.mj_step(self.mj_model, self.mj_data)

        # 3. Update Visualizer
        if self.viewer.is_running():
            self.viewer.sync()
        else:
            self.viewer.close()

        # 4. Retrieve New State
        remote_q = self.mj_data.qpos[:self.n_joints].copy()
        remote_qd = self.mj_data.qvel[:self.n_joints].copy()

        # 5. LOGGING (Modified)
        if self.local_state_received:
            tracking_error = np.linalg.norm(remote_q - self.local_q)
            
            # Configure numpy print options for readability
            np.set_printoptions(precision=3, suppress=True, linewidth=200)
            
            self.get_logger().info(
                f"\n=== CONTROL LOOP DEBUG ===\n"
                f"Desired Q (Leader): {self.local_q}\n"
                f"True Q (Remote):    {remote_q}\n"
                f"RL Tau (Action):    {applied_tau}\n"
                f"L2 Error:           {tracking_error:.4f}\n"
                f"==========================",
                throttle_duration_sec=0.5
            )

        # 6. Publish Joint States
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.name = self.joint_names
        msg.position = remote_q.tolist()
        msg.velocity = remote_qd.tolist()
        msg.effort = applied_tau.tolist() # Publish the torque we actually applied
        self.joint_state_pub.publish(msg)

        # 7. Publish EE Pose
        ee_id = self.mj_model.body(cfg.EE_BODY_NAME).id
        ee_pos = self.mj_data.xpos[ee_id]
        
        ee_msg = PointStamped()
        ee_msg.header = msg.header
        ee_msg.point.x = float(ee_pos[0])
        ee_msg.point.y = float(ee_pos[1])
        ee_msg.point.z = float(ee_pos[2])
        self.ee_pose_pub.publish(ee_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RemoteRobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'viewer') and node.viewer.is_running():
            node.viewer.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()