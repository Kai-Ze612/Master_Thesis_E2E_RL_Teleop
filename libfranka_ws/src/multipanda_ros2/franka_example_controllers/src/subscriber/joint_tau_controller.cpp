#include <franka_example_controllers/subscriber/joint_tau_controller.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <string>
#include <Eigen/Eigen>

namespace franka_example_controllers {

controller_interface::InterfaceConfiguration
JointTauController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
JointTauController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
 
  // Joint position and velocity interfaces
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
 
  // Robot model interfaces for dynamics
  for (const auto& franka_robot_model_name : franka_robot_model_->get_state_interface_names()) {
    config.names.push_back(franka_robot_model_name);
  }
 
  return config;
}

controller_interface::return_type
JointTauController::update(const rclcpp::Time& time, const rclcpp::Duration& period) {
  // Update joint states
  updateJointStates();
  Vector7d tau_commanded = tau_d_;
  // Check if joint states are valid (not NaN or infinite)
  bool joint_states_valid = q_.allFinite() && dq_.allFinite();
  if (!joint_states_valid) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
                         "Invalid joint states detected, setting torques to zero");
    tau_commanded.setZero();
  }
  // Apply rate limiting
  double dt = period.seconds();
  tau_commanded = applyRateLimiting(tau_commanded, dt);
  // Apply safety limits
  tau_commanded = applySafetyLimits(tau_commanded);
  // Check for command timeout or emergency stop
  if (isCommandTimedOut() || emergency_stop_) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
                         "Command timeout or emergency stop - setting torques to zero");
    tau_commanded.setZero();
  }
  // Log commanded torques for debugging
  RCLCPP_DEBUG_STREAM(get_node()->get_logger(), "[" << time.seconds() << "] Joint positions: " << q_.transpose());
  RCLCPP_DEBUG_STREAM(get_node()->get_logger(), "[" << time.seconds() << "] Desired torques: " << tau_d_.transpose());
  RCLCPP_DEBUG_STREAM(get_node()->get_logger(), "[" << time.seconds() << "] Commanded torques: " << tau_commanded.transpose());
  // Send commands to hardware
  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_commanded(i));
  }
  // Store previous torques for rate limiting
  tau_d_prev_ = tau_commanded;
  return controller_interface::return_type::OK;
}

CallbackReturn
JointTauController::on_init() {
  try {
    // Declare parameters
    auto_declare<std::string>("arm_id", "panda");
    auto_declare<std::vector<double>>("tau_max", {87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0});
    auto_declare<std::vector<double>>("tau_min", {-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0});
    auto_declare<double>("torque_rate_limit", 1000.0); // Nm/s
    auto_declare<double>("command_timeout", 0.5); // Increased to avoid timeouts
    // Create subscription for desired torques
    sub_desired_torques_ = get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/joint_tau/torques_desired", 1,
      std::bind(&JointTauController::desiredTorqueCallback, this, std::placeholders::_1)
    );
    RCLCPP_INFO(get_node()->get_logger(), "Joint Tau Controller initialized successfully");
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
 
  return CallbackReturn::SUCCESS;
}

CallbackReturn
JointTauController::on_configure(const rclcpp_lifecycle::State& /*previous_state*/) {
  // Get parameters
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  auto tau_max = get_node()->get_parameter("tau_max").as_double_array();
  auto tau_min = get_node()->get_parameter("tau_min").as_double_array();
  torque_rate_limit_ = get_node()->get_parameter("torque_rate_limit").as_double();
  command_timeout_ = get_node()->get_parameter("command_timeout").as_double();
  // Initialize robot model
  franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
      franka_semantic_components::FrankaRobotModel(arm_id_ + "/robot_model", arm_id_));
  // Validate parameters
  if (tau_max.size() != static_cast<size_t>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "tau_max should be of size %d but is of size %zu", num_joints, tau_max.size());
    return CallbackReturn::FAILURE;
  }
 
  if (tau_min.size() != static_cast<size_t>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "tau_min should be of size %d but is of size %zu", num_joints, tau_min.size());
    return CallbackReturn::FAILURE;
  }
  // Set torque limits
  for (int i = 0; i < num_joints; ++i) {
    tau_max_(i) = tau_max[i];
    tau_min_(i) = tau_min[i];
  }
  // Initialize control variables
  tau_d_.setZero();
  tau_d_prev_.setZero();
  emergency_stop_ = false;
  RCLCPP_INFO(get_node()->get_logger(),
              "Joint Tau Controller configured for arm: %s", arm_id_.c_str());
  return CallbackReturn::SUCCESS;
}

CallbackReturn
JointTauController::on_activate(const rclcpp_lifecycle::State& /*previous_state*/) {
  // Update joint states and assign robot model interfaces
  updateJointStates();
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  // Initialize desired torques to zero
  tau_d_.setZero();
  tau_d_prev_.setZero();
 
  // Reset emergency stop and timer
  emergency_stop_ = false;
  first_command_received_ = false;
  last_command_time_ = get_node()->get_clock()->now();
  RCLCPP_INFO(get_node()->get_logger(), "Joint Tau Controller activated");
 
  return CallbackReturn::SUCCESS;
}

CallbackReturn
JointTauController::on_deactivate(const rclcpp_lifecycle::State& /*previous_state*/) {
  // Set all torques to zero for safety
  for (size_t i = 0; i < command_interfaces_.size(); ++i) {
    command_interfaces_[i].set_value(0.0);
  }
 
  RCLCPP_INFO(get_node()->get_logger(), "Joint Tau Controller deactivated");
 
  return CallbackReturn::SUCCESS;
}

void JointTauController::updateJointStates() {
  for (int i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);
    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");
    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
    RCLCPP_DEBUG_STREAM(get_node()->get_logger(), "[" << get_node()->get_clock()->now().seconds() << "] Joint " << i+1 << ": pos=" << q_(i) << ", vel=" << dq_(i));
  }
}

void JointTauController::desiredTorqueCallback(const std_msgs::msg::Float64MultiArray& msg) {
  // Validate message size
  if (msg.data.size() != static_cast<size_t>(num_joints)) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Received torque command with %zu elements, expected %d",
                 msg.data.size(), num_joints);
    return;
  }
  // Update desired torques
  for (int i = 0; i < num_joints; ++i) {
    tau_d_(i) = msg.data[i];
  }
  // Update command timestamp
  last_command_time_ = get_node()->get_clock()->now();
 
  // Reset emergency stop if new command received
  emergency_stop_ = false;
  first_command_received_ = true;
  RCLCPP_DEBUG_STREAM(get_node()->get_logger(), "[" << last_command_time_.seconds() << "] Received torque command: " << tau_d_.transpose());
}

JointTauController::Vector7d JointTauController::applySafetyLimits(const Vector7d& tau_commanded) {
  Vector7d tau_limited = tau_commanded;
 
  for (int i = 0; i < num_joints; ++i) {
    // Clamp torques within safety limits
    tau_limited(i) = std::clamp(tau_commanded(i), tau_min_(i), tau_max_(i));
   
    // Warn if limits are exceeded
    if (std::abs(tau_commanded(i) - tau_limited(i)) > 1e-6) {
      RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
                           "Joint %d torque limited: commanded=%.3f, limited=%.3f",
                           i+1, tau_commanded(i), tau_limited(i));
    }
  }
 
  return tau_limited;
}

JointTauController::Vector7d JointTauController::applyRateLimiting(const Vector7d& tau_desired, double dt) {
  Vector7d tau_rate_limited = tau_desired;
 
  if (dt > 0.0) {
    double max_change = torque_rate_limit_ * dt;
   
    for (int i = 0; i < num_joints; ++i) {
      double tau_change = tau_desired(i) - tau_d_prev_(i);
     
      if (std::abs(tau_change) > max_change) {
        tau_rate_limited(i) = tau_d_prev_(i) + std::copysign(max_change, tau_change);
       
        RCLCPP_DEBUG_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
                             "Joint %d torque rate limited: desired_change=%.3f, max_change=%.3f",
                             i+1, tau_change, max_change);
      }
    }
  }
 
  return tau_rate_limited;
}

bool JointTauController::isCommandTimedOut() const {
  auto current_time = get_node()->get_clock()->now();
  auto time_since_command = (current_time - last_command_time_).seconds();
  return time_since_command > command_timeout_;
}

} // namespace franka_example_controllers

#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::JointTauController,
                       controller_interface::ControllerInterface)