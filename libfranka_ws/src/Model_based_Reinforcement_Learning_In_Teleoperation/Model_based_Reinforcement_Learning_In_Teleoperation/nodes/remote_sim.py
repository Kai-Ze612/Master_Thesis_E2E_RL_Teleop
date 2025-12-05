"""
Unified Teleoperation Node - Optimized O(1) Inference.
- Maintains a "Real-Time" internal state.
- Steps forward ONCE per control cycle (no loops).
- Constant inference time (~2ms) regardless of delay.
"""

import numpy as np
import torch
import time
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped

import mujoco

from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import DelaySimulator, ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent_autoregressive.sac_policy_network import StateEstimator, Actor
import Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config as cfg

class UnifiedTeleopNode(Node):
    def __init__(self):
        super().__init__('unified_teleop_node')

        # 1. SETUP
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_fp16_ = (self.device_.type == 'cuda')
        
        self.declare_parameter('experiment_config', ExperimentConfig.LOW_DELAY.value)
        self.delay_config_ = ExperimentConfig(self.get_parameter('experiment_config').value)
        self.delay_simulator_ = DelaySimulator(cfg.DEFAULT_CONTROL_FREQ, self.delay_config_, seed=50)

        # Load Models
        self.state_estimator_ = StateEstimator().to(self.device_)
        self.actor_ = Actor(state_dim=cfg.OBS_DIM).to(self.device_)
        self._load_models()
        if self.use_fp16_:
            self.state_estimator_.half()
            self.actor_.half()
        self._warmup_models()

        # 2. BUFFERS & STATE
        self.leader_q_history_ = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.leader_qd_history_ = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
        self.remote_q_history_ = deque(maxlen=cfg.REMOTE_HISTORY_LEN)
        self.remote_qd_history_ = deque(maxlen=cfg.REMOTE_HISTORY_LEN)
        
        # Pre-fill Remote History
        for _ in range(cfg.REMOTE_HISTORY_LEN):
            self.remote_q_history_.append(np.zeros(cfg.N_JOINTS, dtype=np.float32))
            self.remote_qd_history_.append(np.zeros(cfg.N_JOINTS, dtype=np.float32))

        self.pending_leader_packets_ = deque() 
        self.action_delay_queue_ = deque(maxlen=cfg.DEPLOYMENT_HISTORY_BUFFER_SIZE)
        
        self.is_leader_ready_ = False
        self.warmup_steps_ = int(cfg.WARM_UP_DURATION * cfg.DEFAULT_CONTROL_FREQ)
        self.warmup_steps_count_ = 0
        self.prediction_ema_ = None
        
        # --- OPTIMIZATION: RUNNING STATE ---
        # This state represents "NOW" (or slightly future if compensating action delay).
        # We update it incrementally (1 step per tick).
        self.rt_hidden_state_ = None
        self.rt_input_norm_ = None
        
        # 3. SIMULATION & ROS
        self.model = mujoco.MjModel.from_xml_path(cfg.DEFAULT_MUJOCO_MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.ee_body_id = self.model.body(cfg.EE_BODY_NAME).id
        self.data.qpos[:cfg.N_JOINTS] = cfg.INITIAL_JOINT_CONFIG
        mujoco.mj_step(self.model, self.data)
        
        self.control_freq_ = cfg.DEFAULT_CONTROL_FREQ
        self.dt_ = 1.0 / self.control_freq_
        self.target_joint_names_ = [f'panda_joint{i+1}' for i in range(cfg.N_JOINTS)]

        self.create_subscription(JointState, 'local_robot/joint_states', self.local_robot_state_callback, 10)
        self.sim_state_pub_ = self.create_publisher(JointState, 'remote_robot/joint_states', 10)
        self.ee_pub_ = self.create_publisher(PointStamped, 'remote_robot/ee_pose', 10)

        self.timer_ = self.create_timer(self.dt_, self.loop_callback)
        self.step_counter_ = 0
        
        self.get_logger().info("UNIFIED NODE (O(1) STREAMING) READY.")

    def _load_models(self):
        try:
            lstm_ckpt = torch.load(cfg.LSTM_MODEL_PATH, map_location=self.device_, weights_only=False)
            if 'state_estimator_state_dict' in lstm_ckpt:
                self.state_estimator_.load_state_dict(lstm_ckpt['state_estimator_state_dict'])
            else:
                self.state_estimator_.load_state_dict(lstm_ckpt)
            self.state_estimator_.eval()
            
            sac_ckpt = torch.load(cfg.RL_MODEL_PATH, map_location=self.device_, weights_only=False)
            self.actor_.load_state_dict(sac_ckpt['actor_state_dict'])
            self.actor_.eval()
        except Exception as e:
            self.get_logger().fatal(f"Model load failed: {e}")
            raise

    def _warmup_models(self):
        dtype = torch.float16 if self.use_fp16_ else torch.float32
        with torch.no_grad():
            dummy = torch.zeros((1, 1, 15), device=self.device_, dtype=dtype)
            self.state_estimator_.lstm(dummy)
            self.state_estimator_.forward_step(dummy, None)

    def _normalize_input(self, q, qd, delay_scalar):
        q_norm = (q - cfg.Q_MEAN) / cfg.Q_STD
        qd_norm = (qd - cfg.QD_MEAN) / cfg.QD_STD
        return np.concatenate([q_norm, qd_norm, [delay_scalar]])

    def _denormalize_output(self, pred_norm):
        q_norm = pred_norm[:7]
        qd_norm = pred_norm[7:]
        q = (q_norm * cfg.Q_STD) + cfg.Q_MEAN
        qd = (qd_norm * cfg.QD_STD) + cfg.QD_MEAN
        return q, qd

    # -------------------------------------------------------------------------
    # CALLBACKS
    # -------------------------------------------------------------------------
    def local_robot_state_callback(self, msg: JointState):
        """Receive Real Leader State -> Push to Queue (Simulate Network)"""
        arrival_time = self.get_clock().now().nanoseconds / 1e9
        try:
            name_to_index = {name: i for i, name in enumerate(msg.name)}
            pos = [msg.position[name_to_index[name]] for name in self.target_joint_names_]
            vel = [msg.velocity[name_to_index[name]] for name in self.target_joint_names_]
            q_new = np.array(pos, dtype=np.float32)
            qd_new = np.array(vel, dtype=np.float32)
            
            if not self.is_leader_ready_:
                # Prefill history
                qd_zero = np.zeros_like(qd_new)
                for _ in range(cfg.RNN_SEQUENCE_LENGTH + 20):
                    self.leader_q_history_.append(q_new.copy())
                    self.leader_qd_history_.append(qd_zero.copy())
                self.is_leader_ready_ = True
                self.get_logger().info("Leader Connected.")
            else:
                self.pending_leader_packets_.append({'q': q_new, 'qd': qd_new, 't': arrival_time})
        except Exception: pass

    # -------------------------------------------------------------------------
    # INFERENCE: O(1) STREAMING
    # -------------------------------------------------------------------------
    def _run_agent_inference(self, rem_q, rem_qd, new_packets, delay_scalar):
        dtype = torch.float16 if self.use_fp16_ else torch.float32
        
        # 1. INITIALIZATION (On First Run Only)
        # We need to sync our "Real-Time Head" to the very first packet we see
        if self.rt_hidden_state_ is None:
            if new_packets == 0:
                # Can't start without data
                return rem_q, rem_qd, np.zeros(cfg.N_JOINTS)
            
            # Use the newest available packet to start the chain
            # Note: This is an approximation. Ideally we would rollout ONCE here.
            # But for startup, taking the packet as "Now" is acceptable to seed the state.
            history_len = len(self.leader_q_history_)
            start_idx = history_len - new_packets
            seq_buffer = []
            for i in range(start_idx, history_len):
                step_vec = self._normalize_input(self.leader_q_history_[i], self.leader_qd_history_[i], delay_scalar)
                seq_buffer.append(step_vec)
            
            seq_t = torch.tensor(np.array(seq_buffer), dtype=dtype, device=self.device_).unsqueeze(0)
            with torch.no_grad():
                lstm_out, self.rt_hidden_state_ = self.state_estimator_.lstm(seq_t)
                last_hidden = lstm_out[:, -1, :]
                vel_pred = self.state_estimator_.fc(last_hidden)
                prev_state = seq_t[:, -1, :14]
                self.rt_input_norm_ = prev_state + vel_pred * self.state_estimator_.dt_scale
        
        # 2. O(1) STEP FORWARD
        # We assume 'rt_input_norm_' is already at (t). We want (t+1).
        # We feed delay=0.0 because we want to predict Real Time (or ActionCompensated Future).
        
        # Note: If we want to compensate Action Delay, we can conceptually say
        # "We are at t, we want t+1".
        # If the Action Delay is constant, this Streaming Head naturally maintains 
        # a "Lead" if it was initialized with a lead. 
        # Here, we simply run it freely. It will track the leader's velocity profile.
        
        target_delay = 0.0 # We want Real-Time Prediction
        delay_t = torch.tensor([[[target_delay]]], dtype=dtype, device=self.device_)
        state_in = self.rt_input_norm_.unsqueeze(1)
        curr_input = torch.cat([state_in, delay_t], dim=2)
        
        with torch.no_grad():
            self.rt_input_norm_, self.rt_hidden_state_ = self.state_estimator_.forward_step(
                curr_input, self.rt_hidden_state_
            )
            self.rt_input_norm_ = self.rt_input_norm_.squeeze(1)

        # 3. RESULT
        pred_q_norm_numpy = self.rt_input_norm_.float().cpu().numpy()[0]
        final_q, final_qd = self._denormalize_output(pred_q_norm_numpy)

        # EMA
        if self.prediction_ema_ is None: self.prediction_ema_ = final_q
        else: self.prediction_ema_ = cfg.PREDICTION_EMA_ALPHA * final_q + (1.0 - cfg.PREDICTION_EMA_ALPHA) * self.prediction_ema_
        final_q = self.prediction_ema_

        # 4. RL ACTION
        self.remote_q_history_.append(rem_q.copy())
        self.remote_qd_history_.append(rem_qd.copy())
        
        obs_vec = np.concatenate([
            rem_q, rem_qd, 
            np.concatenate(list(self.remote_q_history_)), 
            np.concatenate(list(self.remote_qd_history_)),
            final_q, final_qd, 
            (final_q - rem_q), (final_qd - rem_qd),
            [0.0]
        ]).astype(np.float32)
        
        actor_input_t = torch.tensor(obs_vec, dtype=dtype, device=self.device_).unsqueeze(0)
        with torch.no_grad():
            action_t, _, _ = self.actor_.sample(actor_input_t, deterministic=True)
        
        return final_q, final_qd, action_t.float().cpu().numpy().flatten()

    def _update_history_from_pending(self):
        if not self.pending_leader_packets_: return 0, 0.0
        
        history_len = len(self.leader_q_history_)
        obs_delay_steps = self.delay_simulator_.get_observation_delay_steps(history_len)
        obs_delay_sec = obs_delay_steps * self.dt_
        norm_delay_scalar = float(obs_delay_steps) / cfg.DELAY_INPUT_NORM_FACTOR
        
        now = self.get_clock().now().nanoseconds / 1e9
        new_packets_count = 0
        
        while self.pending_leader_packets_:
            packet = self.pending_leader_packets_[0]
            if (now - packet['t']) >= obs_delay_sec:
                p = self.pending_leader_packets_.popleft()
                self.leader_q_history_.append(p['q'])
                self.leader_qd_history_.append(p['qd'])
                new_packets_count += 1
            else:
                break
        return new_packets_count, norm_delay_scalar

    # -------------------------------------------------------------------------
    # MAIN LOOP & PHYSICS
    # -------------------------------------------------------------------------
    def loop_callback(self):
        if not self.is_leader_ready_: return
        self.step_counter_ += 1
        start_time = time.perf_counter()
        
        if self.warmup_steps_count_ < self.warmup_steps_:
            self.warmup_steps_count_ += 1
            # Simple Warmup
            current_remote_q = self.data.qpos[:cfg.N_JOINTS].copy()
            target = self.leader_q_history_[-1]
            alpha = self.warmup_steps_count_ / self.warmup_steps_
            self._apply_control((1-alpha)*current_remote_q + alpha*target, np.zeros(7), np.zeros(7))
            self._publish_sim_state()
            return

        # 1. READ STATE
        q_rem = self.data.qpos[:cfg.N_JOINTS].copy().astype(np.float32)
        qd_rem = self.data.qvel[:cfg.N_JOINTS].copy().astype(np.float32)

        # 2. UPDATE BUFFER (We still consume buffers to keep history valid, but we don't sync to them)
        new_cnt, delay_s = self._update_history_from_pending()

        # 3. PREDICT (STREAMING O(1))
        target_q, target_qd, tau_rl = self._run_agent_inference(q_rem, qd_rem, new_cnt, delay_s)

        # 4. ACTION DELAY QUEUE
        self.action_delay_queue_.append({'q': target_q, 'qd': target_qd, 'tau': tau_rl})
        act_delay_steps = self.delay_simulator_.get_action_delay_steps()
        
        if act_delay_steps >= len(self.action_delay_queue_):
            cmd = {'q': cfg.INITIAL_JOINT_CONFIG, 'qd': np.zeros(7), 'tau': np.zeros(7)}
        else:
            cmd = self.action_delay_queue_[-1 - act_delay_steps]

        # 5. CONTROL & STEP
        self._apply_control(cmd['q'], cmd['qd'], cmd['tau'])
        
        # 6. LOGGING
        self._publish_sim_state()
        if self.step_counter_ % 20 == 0:
            if self.pending_leader_packets_: true_ref = self.pending_leader_packets_[-1]['q']
            else: true_ref = self.leader_q_history_[-1]
            
            err = np.linalg.norm(q_rem - true_ref)
            inf_time = (time.perf_counter() - start_time) * 1000.0
            self.get_logger().info(f"[Step {self.step_counter_}] TrueError: {err:.4f} | InfTime: {inf_time:.2f}ms")

    def _apply_control(self, q_des, qd_des, tau_rl):
        q = self.data.qpos[:cfg.N_JOINTS].copy()
        qd = self.data.qvel[:cfg.N_JOINTS].copy()
        q_err = q_des - q
        qd_err = qd_des - qd
        acc_des = cfg.DEFAULT_KP_REMOTE * q_err + cfg.DEFAULT_KD_REMOTE * qd_err
        
        self.data.qpos[:cfg.N_JOINTS] = q
        self.data.qvel[:cfg.N_JOINTS] = qd
        self.data.qacc[:cfg.N_JOINTS] = acc_des
        mujoco.mj_inverse(self.model, self.data)
        tau_id = self.data.qfrc_inverse[:cfg.N_JOINTS].copy()
        
        self.data.qpos[:cfg.N_JOINTS] = q
        self.data.qvel[:cfg.N_JOINTS] = qd
        
        tau_total = tau_id + tau_rl
        tau_total = np.clip(tau_total, -cfg.TORQUE_LIMITS, cfg.TORQUE_LIMITS)
        tau_total[-1] = 0.0
        
        self.data.ctrl[:cfg.N_JOINTS] = tau_total
        mujoco.mj_step(self.model, self.data)

    def _publish_sim_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.target_joint_names_
        msg.position = self.data.qpos[:cfg.N_JOINTS].tolist()
        msg.velocity = self.data.qvel[:cfg.N_JOINTS].tolist()
        self.sim_state_pub_.publish(msg)
        
        ee_pos = self.data.xpos[self.ee_body_id]
        pt = PointStamped()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = "world"
        pt.point.x, pt.point.y, pt.point.z = float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])
        self.ee_pub_.publish(pt)

def main(args=None):
    rclpy.init(args=args)
    node = UnifiedTeleopNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()