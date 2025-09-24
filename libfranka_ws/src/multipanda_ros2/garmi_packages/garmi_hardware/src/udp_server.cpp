#include <garmi_hardware/udp_server.hpp>

namespace garmi_hardware{
    
    // Publics
    UDPServer::UDPServer(HardwareID hw_id){
        id_ = hw_id;
        // grab the IP and the port from the environmental variables
        const char* vm_ip = std::getenv(env_ip_[id_].c_str());
        const char* vm_send_port_char = std::getenv(env_send_port_[id_].c_str());
        const char* vm_recv_port_char = std::getenv(env_recv_port_[id_].c_str());
        if(!vm_ip || !vm_send_port_char || !vm_recv_port_char){
            std::string error_msg = "One or more of the environment variables are not set! Make sure the following environment variables exist:\n" + env_ip_[id_] + ", " + env_send_port_[id_] + ", " + env_recv_port_[id_];
            throw std::runtime_error(error_msg);
        }
        vm_send_port_ = static_cast<uint16_t>(std::stoi(vm_send_port_char));
        vm_recv_port_ = static_cast<uint16_t>(std::stoi(vm_recv_port_char));

        // Setting up the IP and Port for sending the UDP packet to the VM
        memset(&vm_addr_, 0, sizeof(vm_addr_));
        vm_addr_.sin_family = AF_INET;
        vm_addr_.sin_port = htons(vm_send_port_);
        vm_addr_.sin_addr.s_addr = inet_addr(vm_ip);
        
        // Setting up the Port for receiving the UDP packet from the VM
        memset(&local_addr_, 0, sizeof(local_addr_));
        local_addr_.sin_addr.s_addr = INADDR_ANY;
        local_addr_.sin_port = htons(vm_recv_port_);
        
        // Resize the buffer to the size of struct
        send_buffer_.resize(sizeof(FrankaMotorCommand));
        recv_buffer_.resize(sizeof(FrankaMotorState));
    };

    void UDPServer::initialize_udp(){
        // Create the sockets
        if ((send_sockfd_ = socket(AF_INET, SOCK_DGRAM, 0)) < 0){
            throw std::runtime_error("Socket creation failed for send_sockfd_!");
        }
        if ((recv_sockfd_ = socket(AF_INET, SOCK_DGRAM, 0)) < 0){
            throw std::runtime_error("Socket creation failed for recv_sockfd_!");
        }

        // Bind the recv socket
        if (bind(recv_sockfd_, (const struct sockaddr *)&local_addr_, sizeof(local_addr_)) < 0){
            throw std::runtime_error(std::string("Binding to the receiver socket failed at port: " 
                                                    + std::to_string(vm_recv_port_)));
        }

        // Now the ports should be ready for use
    }
    void UDPServer::shutdown_udp(){
        close(send_sockfd_);
        close(recv_sockfd_);
    }

    bool UDPServer::send_udp_packet(){
        serializeFrankaMotorCommand();
        std::lock_guard<std::mutex> lock(state_mtx_);
        sendto(send_sockfd_, send_buffer_.data(), sizeof(send_buffer_), MSG_CONFIRM, (const struct sockaddr *)&vm_addr_,
           sizeof(vm_addr_));
        return true;
    };

    bool UDPServer::recv_udp_packet(){
        {   // scope for lock_guard
            std::lock_guard<std::mutex> lock(cmd_mtx_);
            socklen_t len;
            len = sizeof(local_addr_); // len is value/resuslt
            recvfrom(recv_sockfd_, recv_buffer_.data(), sizeof(recv_buffer_), MSG_WAITALL, (struct sockaddr *)&local_addr_, &len);
        }
        deserializeFrankaMotorState();
        return true;
    };
    
    FrankaMotorState UDPServer::get_motor_state(){
        std::lock_guard<std::mutex> lock(state_mtx_);
        return motor_state_;
    }
    void UDPServer::set_motor_command(FrankaMotorCommand cmd){
        std::lock_guard<std::mutex> lock(cmd_mtx_);
        motor_cmd_ = cmd;
    }
    // Privates
    void UDPServer::deserializeFrankaMotorState() {
        if (recv_buffer_.size() != sizeof(FrankaMotorState)) {
            throw std::runtime_error("Invalid buffer size for FrankaMotorState");
        }
        std::lock_guard<std::mutex> lock(state_mtx_);
        std::memcpy(&motor_state_, recv_buffer_.data(), sizeof(FrankaMotorState));
    }

    void UDPServer::serializeFrankaMotorCommand() {
        // std::vector<uint8_t> buffer(sizeof(FrankaMotorCommand));
        if (send_buffer_.size() != sizeof(FrankaMotorCommand)) {
            throw std::runtime_error("Invalid buffer size for FrankaMotorCommand");
        }
        std::lock_guard<std::mutex> lock(cmd_mtx_);
        std::memcpy(send_buffer_.data(), &motor_cmd_, sizeof(FrankaMotorCommand));
    }
}