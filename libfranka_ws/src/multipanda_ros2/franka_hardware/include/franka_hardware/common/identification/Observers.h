#ifndef OBSERVERS_H_   /* Include guard */
#define OBSERVERS_H_

#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include "Dynamics.h"
#include <franka/robot.h>
#include <franka/model.h>
#include <franka_hardware/common/model_base.hpp>
#include <array>

using namespace Eigen;

namespace panda_identification {
  class Observers {
    public:
      Observers(std::shared_ptr<Dynamics> dyn);
      ~Observers();
      
      VectorXd get_tau_ext_hat_filtered(const franka::RobotState& state);
      // VectorXd get_O_F_ext_hat_K(const franka::RobotState& state, const franka::Model& model);
      // xx_F_ext_hat_K acts as the impl function; O_F and K_F are interface functions that just call xx_F with the right argument
      VectorXd get_O_F_ext_hat_K(const franka::RobotState& state, const franka_hardware::ModelBase* model);
      // VectorXd get_xx_F_ext_hat_K(const franka::RobotState& state, const franka::Model& model, franka::Frame frame);
      // TODO: Clarify with Mario if this is needed
      // VectorXd get_K_com_ext_K(const franka::RobotState& state, const franka::Model& model, VectorXd O_gravity);
      // VectorXd get_O_F_ext_hat_K(const franka::RobotState& state, std::array<double, 42>& zeroJacobian_array);
      VectorXd get_K_F_ext_hat_K(const franka::RobotState& state, std::array<double, 42>& k_Jacobian_array);
      VectorXd get_xx_F_ext_hat_K(const franka::RobotState& state, std::array<double, 42>& xx_Jacobian_array);

      // not used; was iplemented for IAM Project
      // double get_obj_mass(const franka::RobotState& state, std::array<double, 42>& zeroJacobian_array);
      // double get_ext_mass(const franka::RobotState& state, const franka::Model& model, VectorXd O_gravity);

    private:
      VectorXd _integral;
      VectorXd _r; //residual
      std::shared_ptr<Dynamics> _dyn;
  };
} // namespace panda_identification

#endif // OBSERVERS_H_