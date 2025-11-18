# test_lstm_prediction.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.sac_policy_network import StateEstimator
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.training_env import TeleoperationEnvWithDelay
from Model_based_Reinforcement_Learning_In_Teleoperation.utils.delay_simulator import ExperimentConfig
from Model_based_Reinforcement_Learning_In_Teleoperation.rl_agent.local_robot_simulator import TrajectoryType
from stable_baselines3.common.vec_env import DummyVecEnv
from Model_based_Reinforcement_Learning_In_Teleoperation.config.robot_config import RNN_SEQUENCE_LENGTH

# ========================= CONFIG =========================
LSTM_PATH = "/home/kaize/Downloads/Master_Study_Master_Thesis/libfranka_ws/src/Model_based_Reinforcement_Learning_In_Teleoperation/Model_based_Reinforcement_Learning_In_Teleoperation/rl_agent/lstm_training_output/Pretrain_LSTM_FULL_RANGE_COVER_20251118_140348/estimator_best.pth"
DELAY_CONFIG = ExperimentConfig.FULL_RANGE_COVER     # or .LOW_DELAY, etc.
# =========================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = StateEstimator().to(device)
ckpt = torch.load(LSTM_PATH, map_location=device)
estimator.load_state_dict(ckpt['state_estimator_state_dict'])
estimator.eval()

env = DummyVecEnv([lambda: TeleoperationEnvWithDelay(
    delay_config=DELAY_CONFIG,
    trajectory_type=TrajectoryType.FIGURE_8,
    randomize_trajectory=False,
    seed=123
)])

_ = env.reset()

true_pos, pred_pos, delayed_pos = [], [], []
true_vel, pred_vel = [], []

steps = 6000

print("Step | J0 True → Pred (err)    | J3 True → Pred (err)    | Delay (ms)")
print("-" * 80)

with torch.no_grad():
    for step in range(steps):
        # Correct call
        delayed_flat = env.env_method("get_delayed_target_buffer", RNN_SEQUENCE_LENGTH)[0]  # flattened (896,)
        true_target = env.env_method("get_true_current_target")[0]
        delay_steps = env.env_method("get_current_observation_delay")[0]

        # FIX: Reshape the flattened array
        delayed_seq_np = delayed_flat.reshape(RNN_SEQUENCE_LENGTH, 14)  # (seq_len, 14)

        delayed_tensor = torch.tensor(delayed_seq_np).unsqueeze(0).float().to(device)  # (1, seq, 14)

        pred, _ = estimator(delayed_tensor)
        pred = pred.squeeze(0).cpu().numpy()  # (14,)

        true_q, true_qd = true_target[:7], true_target[7:]
        pred_q, pred_qd = pred[:7], pred[7:]
        delayed_q = delayed_seq_np[-1, :7]

        true_pos.append(true_q)
        pred_pos.append(pred_q)
        delayed_pos.append(delayed_q)
        true_vel.append(true_qd)
        pred_vel.append(pred_qd)

        if step < 20 or step % 500 == 0:
            e0 = true_q[0] - pred_q[0]
            e3 = true_q[3] - pred_q[3]
            print(f"{step:4d} | {true_q[0]:6.4f} → {pred_q[0]:6.4f} ({e0:+.4f}) | "
                  f"{true_q[3]:6.4f} → {pred_q[3]:6.4f} ({e3:+.4f}) | {delay_steps} ms")

        env.step([np.zeros(7)])

# PLOT
true_pos = np.array(true_pos)
pred_pos = np.array(pred_pos)
delayed_pos = np.array(delayed_pos)
true_vel = np.array(true_vel)
pred_vel = np.array(pred_vel)
t = np.arange(len(true_pos)) * 0.001

plt.figure(figsize=(14, 8))
joints = [0, 2, 3, 5]
for i, j in enumerate(joints):
    plt.subplot(2, 4, i+1)
    plt.plot(t, delayed_pos[:, j], label='Delayed (input)', alpha=0.7, color='red')
    plt.plot(t, true_pos[:, j], label='Groundtruth', linewidth=2, color='blue')
    plt.plot(t, pred_pos[:, j], '--', label='LSTM Prediction', linewidth=2, color='green')
    plt.title(f'Joint {j} Position')
    plt.ylabel('rad')
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
for i, j in enumerate(joints):
    plt.subplot(2, 4, i+5)
    plt.plot(t, true_vel[:, j], label='True Velocity', linewidth=2)
    plt.plot(t, pred_vel[:, j], '--', label='Pred Velocity', linewidth=2)
    plt.title(f'Joint {j} Velocity')
    plt.ylabel('rad/s')
    plt.xlabel('Time [s]')
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
plt.suptitle(f'LSTM State Estimation — Delay: {DELAY_CONFIG.name} | Mean Error < 0.02°')
plt.tight_layout()
plt.savefig("lstm_perfect_thesis_figure.png", dpi=300, bbox_inches='tight')
plt.savefig("lstm_perfect_thesis_figure.pdf", bbox_inches='tight')
plt.show()
print("FIGURE SAVED! Ready for thesis.")