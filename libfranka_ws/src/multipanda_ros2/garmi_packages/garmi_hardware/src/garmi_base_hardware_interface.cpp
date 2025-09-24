#include <garmi_hardware/garmi_base_hardware_interface.hpp>

namespace garmi_hardware{

using StateInterface = hardware_interface::StateInterface;
using CommandInterface = hardware_interface::CommandInterface;

hardware_interface::CallbackReturn GarmiBaseHardwareInterface::on_init(
    const hardware_interface::HardwareInfo & info){
    
    if (hardware_interface::SystemInterface::on_init(info) != CallbackReturn::SUCCESS) {
        return CallbackReturn::ERROR;
    }

    //====== Checking that the garmi_head.ros2_control.xacro file is configured correctly =====//
    // there should be exactly 2 joints
    if (info_.joints.size() != kNumberOfJoints) {
        RCLCPP_FATAL(getLogger(), "Got %ld joints. Expected %ld.", info_.joints.size(),
                        kNumberOfJoints);
        return CallbackReturn::ERROR;
    }

    // Now loop through all the joints
    for (const auto& joint : info_.joints) {
        // Check number of command interfaces
        if (joint.command_interfaces.size() != 1) {
        RCLCPP_FATAL(getLogger(), "Joint '%s' has %ld command interfaces found. 1 expected.",
                    joint.name.c_str(), joint.command_interfaces.size());
        return CallbackReturn::ERROR;
        }
        
        // Check that the interfaces are named correctly
        //=> Head only has HW_IF_EFFORT; base would only have HW_IF_VELOCITY
        for (const auto & cmd_interface : joint.command_interfaces){
            if (cmd_interface.name != hardware_interface::HW_IF_VELOCITY){   // Effort "effort"
                RCLCPP_FATAL(getLogger(), "Joint '%s' has unexpected command interface '%s'",
                            joint.name.c_str(), cmd_interface.name.c_str());
                return CallbackReturn::ERROR;
            }
        }

        // Check number of state interfaces
        if (joint.state_interfaces.size() != 2) {
            RCLCPP_FATAL(getLogger(), "Joint '%s' has %zu state interfaces found. 2 expected.",
                        joint.name.c_str(), joint.state_interfaces.size());
            return CallbackReturn::ERROR;
        }

        // The 3 interfaces should be Position, Velocity, Effort
        //=> base should have only 2: Position, Velocity
        //=> the order in which you write down the interface in hardware.ros2_control.xacro matters!
        //=> There's definitely a better way to do it, but it's how I wrote it...
        if (joint.state_interfaces[0].name != hardware_interface::HW_IF_POSITION) {
            RCLCPP_FATAL(getLogger(), "Joint '%s' has unexpected state interface '%s'. Expected '%s'",
                        joint.name.c_str(), joint.state_interfaces[0].name.c_str(),
                        hardware_interface::HW_IF_POSITION);
            return CallbackReturn::ERROR;
        }
        if (joint.state_interfaces[1].name != hardware_interface::HW_IF_VELOCITY) {
            RCLCPP_FATAL(getLogger(), "Joint '%s' has unexpected state interface '%s'. Expected '%s'",
                        joint.name.c_str(), joint.state_interfaces[1].name.c_str(),
                        hardware_interface::HW_IF_VELOCITY);
            return CallbackReturn::ERROR;
        }
    }
    
    //====== Create a pointer of udp_server =====//
    udp_server_ = std::make_unique<UDPServer>(hw_id_);
    motor_command_.id_ = hw_id_;
    return CallbackReturn::SUCCESS;
};
    
hardware_interface::CallbackReturn GarmiBaseHardwareInterface::on_configure(
    const rclcpp_lifecycle::State & /*previous_state*/){

    // Activate and deactivate should already do what we need
    RCLCPP_INFO(getLogger(), "on_configure finished");
    return CallbackReturn::SUCCESS;        
};
    
hardware_interface::CallbackReturn GarmiBaseHardwareInterface::on_activate(
    const rclcpp_lifecycle::State & /*previous_state*/){
    
    udp_server_->initialize_udp();
    RCLCPP_INFO(getLogger(), "on_activate finished");
    return CallbackReturn::SUCCESS;
};
        
hardware_interface::CallbackReturn GarmiBaseHardwareInterface::on_deactivate(
    const rclcpp_lifecycle::State & /*previous_state*/){
    
    udp_server_->shutdown_udp();
    RCLCPP_INFO(getLogger(), "on_deactivate finished");
    return CallbackReturn::SUCCESS;
};
            
hardware_interface::return_type GarmiBaseHardwareInterface::read(
    const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/){
    // technically, running at 1khz, this would suffer from minimal hang...
    udp_server_->recv_udp_packet(); 
    motor_state_ = udp_server_->get_motor_state();
    // here, you can also do some stuff with motor_state_.error_
    // and a check for motor_state_.id_, to check that the message is really from head
    // and a check for motor_state_.timestamp_, to check that the dt is within expectation
    //=> For base, you would only write to hw_velocities_.

    hw_velocities_[0] = motor_state_.l_velocity_;
    hw_velocities_[1] = motor_state_.r_velocity_;

    hw_positions_[0] = motor_state_.l_position_;
    hw_positions_[1] = motor_state_.r_position_;
    return hardware_interface::return_type::OK;               
    
};
                
hardware_interface::return_type GarmiBaseHardwareInterface::write(
    const rclcpp::Time & time, const rclcpp::Duration & /*period*/){
    //=> For base, you would write from hw_commands_joint_velocity_.
    motor_command_.l_cmd_ = hw_commands_joint_velocity_[0];
    motor_command_.r_cmd_ = hw_commands_joint_velocity_[1];
    motor_command_.timestamp_ = time.seconds();
    // again, you can add some logic for determining error of some kind.
    // doesn't necessarily have to use franka_joint_driver::Error. Could just be uint8,
    // though then you should update the struct accordingly.
    motor_command_.error_ = franka_joint_driver::Error::kNoError;
    udp_server_->set_motor_command(motor_command_);
    udp_server_->send_udp_packet();
    return hardware_interface::return_type::OK;               
};

std::vector<hardware_interface::StateInterface> GarmiBaseHardwareInterface::export_state_interfaces(){

    std::vector<StateInterface> state_interfaces;
    for(auto i = 0U; i < info_.joints.size(); i++) {
        state_interfaces.emplace_back(StateInterface(
            info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_positions_.at(i)));
        state_interfaces.emplace_back(StateInterface(
            info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_velocities_.at(i)));
    }
    return state_interfaces;
};

std::vector<hardware_interface::CommandInterface> GarmiBaseHardwareInterface::export_command_interfaces(){

    std::vector<CommandInterface> command_interfaces;
    for (auto i = 0U; i < info_.joints.size(); i++) {
        command_interfaces.emplace_back(CommandInterface( // JOINT VELOCITY
            info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_commands_joint_velocity_.at(i)));
    }
    return command_interfaces;
};

} // namespace garmi_hardware

#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(garmi_hardware::GarmiBaseHardwareInterface,
                       hardware_interface::SystemInterface)