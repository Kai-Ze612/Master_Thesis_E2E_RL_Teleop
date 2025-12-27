#pragma once

#include <iostream>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <vector>
#include <cstdlib> // for getenv
#include <mutex>
#include <string>

namespace franka_joint_driver { // directly copied from driver.h
enum class Error{
    kNoError,                 // No error
    kHardwareMismatch,        // Driver does not match the hardware
    kCommunicationError,      // Error in communcation
    kControllerNotSupported,
};
}

namespace garmi_hardware {

enum HardwareID{
    Head = 0,
    Base
};

struct FrankaMotorState{
    HardwareID id_;
    franka_joint_driver::Error error_;
    double timestamp_;
    
    double l_position_;
    double l_velocity_;
    double l_effort_; 
    
    double r_position_;
    double r_velocity_;
    double r_effort_;
};

struct FrankaMotorCommand{
    HardwareID id_;
    franka_joint_driver::Error error_;
    double timestamp_;

    double l_cmd_;
    double r_cmd_;
};

class UDPServer{
public:
    UDPServer(HardwareID hw_id);
    void initialize_udp();
    void shutdown_udp();
    
    bool send_udp_packet();
    bool recv_udp_packet();

    FrankaMotorState get_motor_state();
    void set_motor_command(FrankaMotorCommand cmd);

private:

    void deserializeFrankaMotorState();
    void serializeFrankaMotorCommand();

    // locks
    std::mutex state_mtx_;
    std::mutex cmd_mtx_;

    // Fixed env var names
    std::vector<std::string> env_ip_ = {"HEAD_VM_IP", "BASE_VM_IP"};
    std::vector<std::string> env_send_port_ = {"HEAD_VM_SEND_PORT", "BASE_VM_SEND_PORT"};
    std::vector<std::string> env_recv_port_ = {"HEAD_VM_RECV_PORT", "BASE_VM_RECV_PORT"};

    // UDP-related variables
    int send_sockfd_, recv_sockfd_;
    uint16_t vm_send_port_, vm_recv_port_;
    struct sockaddr_in local_addr_, vm_addr_;
    std::vector<uint8_t> recv_buffer_;
    std::vector<uint8_t> send_buffer_;
    
    // robot control-related variables
    FrankaMotorState motor_state_;
    FrankaMotorCommand motor_cmd_;
    HardwareID id_;
    franka_joint_driver::Error error_;
    double timestamp_;
};

}