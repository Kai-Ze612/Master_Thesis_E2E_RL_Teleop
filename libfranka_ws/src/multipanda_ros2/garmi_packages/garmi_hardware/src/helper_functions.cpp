#include "garmi_hardware/helper_functions.hpp"
#include <string>

namespace garmi_hardware{

// Function for extracting joint number
int get_joint_no(std::string const& s){
    int no = s.back() - '0' - 1;
    return no;
};

} // namespace garmi_hardware
