#pragma once

#include <memory>
#include <string>
#include <vector>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include "rclcpp/clock.hpp"
#include "rclcpp/duration.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "rclcpp/rclcpp.hpp"


#include <hardware_interface/visibility_control.h>
#include <hardware_interface/hardware_info.hpp>
#include <hardware_interface/system_interface.hpp>
#include <hardware_interface/types/hardware_interface_return_values.hpp>

#include "garmi_hardware/udp_server.hpp"
#include "garmi_hardware/helper_functions.hpp"

namespace garmi_hardware{

class GarmiHeadHardwareInterface : public hardware_interface::SystemInterface{

public:
    hardware_interface::CallbackReturn on_init(
        const hardware_interface::HardwareInfo & info) override;
        
    hardware_interface::CallbackReturn on_configure(
        const rclcpp_lifecycle::State & previous_state) override;
        
    hardware_interface::CallbackReturn on_activate(
        const rclcpp_lifecycle::State & previous_state) override;
            
    hardware_interface::CallbackReturn on_deactivate(
        const rclcpp_lifecycle::State & previous_state) override;
                
    hardware_interface::return_type read(
        const rclcpp::Time & time, const rclcpp::Duration & period) override;
                    
    hardware_interface::return_type write(
        const rclcpp::Time & time, const rclcpp::Duration & period) override;

    std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
    std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

private:
    std::unique_ptr<UDPServer> udp_server_;
    FrankaMotorCommand motor_command_;
    FrankaMotorState motor_state_;
    HardwareID hw_id_ = HardwareID::Head;

    //=> values exposed to the controllers.
    //=> for garmi_base, it would be hw_commands_joint_velocity_ and hw_velocities_, hw_positions_
    // Command variable
    std::array<double, 2> hw_commands_joint_effort_{0, 0};
    // State variables
    std::array<double, 2> hw_efforts_{0, 0};
    std::array<double, 2> hw_positions_{0, 0};
    std::array<double, 2> hw_velocities_{0, 0};

    static const size_t kNumberOfJoints = 2;

    rclcpp::Logger getLogger() {
        return rclcpp::get_logger("GarmiHeadHardwareInterface");
    }
};
} // namespace garmi_hardware