#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;

class CartesianImpedancePublisher : public rclcpp::Node {
 public:
  CartesianImpedancePublisher() : Node("cartesian_impedance_publisher") {
    publisher_ =
        this->create_publisher<geometry_msgs::msg::PoseStamped>("/panda/panda_cartesian_impedance_controller/target_pose", 10);

    // Use a timer to publish the message every 100ms.
    timer_ = this->create_wall_timer(
        100ms, std::bind(&CartesianImpedancePublisher::publish_message, this));
  }

 private:
  void publish_message() {
    auto message = std::make_unique<geometry_msgs::msg::PoseStamped>();
    message->header.stamp = this->now();
    message->header.frame_id = "panda_link0"; 
    
    // Set a target pose.
    message->pose.position.x = 0.3;
    message->pose.position.y = 0.4;
    message->pose.position.z = 0.5;
    message->pose.orientation.x = 1.0;
    message->pose.orientation.y = 0.0;
    message->pose.orientation.z = 0.0;
    message->pose.orientation.w = 0.0;

    publisher_->publish(std::move(message));
  }

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CartesianImpedancePublisher>());
  rclcpp::shutdown();
  return 0;
}