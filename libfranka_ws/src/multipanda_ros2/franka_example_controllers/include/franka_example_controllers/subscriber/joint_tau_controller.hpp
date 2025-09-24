#pragma once
#include <string>
#include <memory>
#include <vector>
#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp>
#include "franka_semantic_components/franka_robot_model.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
namespace franka_example_controllers {
/**
 * @brief Joint Torque Controller for Franka Robot
 *
 * This controller receives desired joint torques and directly commands them to the robot.
 * It provides safety mechanisms including torque limiting and rate limiting.
 */
class JointTauController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  /**
   * @brief Configure command interfaces (joint efforts)
   */
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  /**
   * @brief Configure state interfaces (joint positions, velocities, and robot model)
   */
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  /**
   * @brief Main control update loop
   * @param time Current time
   * @param period Control period
   * @return Control update status
   */
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  /**
   * @brief Initialize controller parameters and subscribers
   */
  CallbackReturn on_init() override;
  /**
   * @brief Configure controller with parameters from parameter server
   */
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  /**
   * @brief Activate controller and initialize state
   */
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  /**
   * @brief Deactivate controller and reset commanded torques
   */
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;
 private:
  // Robot configuration
  std::string arm_id_;
  static constexpr int num_joints = 7;
  // Robot model interface
  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
  // Joint state variables
  Vector7d q_; ///< Current joint positions
  Vector7d dq_; ///< Current joint velocities
  Vector7d tau_d_; ///< Desired joint torques
  // Safety parameters
  Vector7d tau_max_; ///< Maximum allowed torques per joint
  Vector7d tau_min_; ///< Minimum allowed torques per joint
  // Control parameters
  double torque_rate_limit_; ///< Maximum rate of torque change [Nm/s]
  Vector7d tau_d_prev_; ///< Previous desired torques for rate limiting
  // ROS interfaces
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_desired_torques_;
  // Safety and monitoring
  rclcpp::Time last_command_time_;
  double command_timeout_; ///< Timeout for torque commands [s]
  bool emergency_stop_; ///< Emergency stop flag
  bool first_command_received_; ///< First command received
  /**
   * @brief Update joint states from hardware interfaces
   */
  void updateJointStates();
  /**
   * @brief Callback for desired joint torques
   * @param msg Array of desired torques for each joint
   */
  void desiredTorqueCallback(const std_msgs::msg::Float64MultiArray& msg);
  /**
   * @brief Apply safety limits to commanded torques
   * @param tau_commanded Input torques to be limited
   * @return Limited torques within safety bounds
   */
  Vector7d applySafetyLimits(const Vector7d& tau_commanded);
  /**
   * @brief Apply torque rate limiting
   * @param tau_desired Desired torques
   * @param dt Time step
   * @return Rate-limited torques
   */
  Vector7d applyRateLimiting(const Vector7d& tau_desired, double dt);
  /**
   * @brief Check if torque command has timed out
   * @return True if command has timed out
   */
  bool isCommandTimedOut() const;
};
} // namespace franka_example_controllers