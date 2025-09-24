/** ------------------------- Revision Code History -------------------
*** Programming Language: C++
*** Description: Panda Dynamics
*** Released Date: Apr. 2023
*** Mario Tröbinger
*** mario.troebinger@tum.de
----------------------------------------------------------------------- */
// #pragma once
#include "franka_hardware/common/identification/Observers.h"


namespace panda_identification {
  Observers::Observers(std::shared_ptr<Dynamics> dyn) {
    _dyn = dyn;
    _integral.setZero(7);
    _r.setZero(7);
  };

  Observers::~Observers(){
  };

  VectorXd Observers::get_tau_ext_hat_filtered(const franka::RobotState& state){
    VectorXd tau_J = Map<VectorXd>(const_cast<double*>(state.tau_J.data()), state.tau_J.size());  
    VectorXd qd = Map<VectorXd>(const_cast<double*>(state.dq.data()), state.dq.size()); // measured torque in Eigen representation
    VectorXd q = Map<VectorXd>(const_cast<double*>(state.q.data()), state.q.size()); // measured torque in Eigen representation

    MatrixXd M = _dyn->get_M(q);
    MatrixXd C = _dyn->get_C(q, qd);
    VectorXd tau_G = _dyn->get_tau_G(q); // gravity torque
    VectorXd qd_zero(7);
    qd_zero.setZero();
    VectorXd tau_Frict = _dyn->get_tau_F(qd_zero); // frictionr torque  
    // VectorXd tau_Frict = _dyn->get_tau_F(qd);

    VectorXd p = M*qd; //Momentum
    double dt = 0.001; // can be read from the stepsize later
    double KO = 10;

    _integral = _integral + (tau_J-tau_G + C.transpose()*qd-tau_Frict+_r)*dt;
    // _integral = _integral + (tau_J-tau_G - C.transpose()*qd+_r)*dt;
    _r = KO * (p - _integral);

    return -_r; // minus to have the same sign like franka
  };
  // TODO: Clarify with mario: O_F or K_F? // guessing it's O_F, since return is O
  // VectorXd Observers::get_O_F_ext_hat_K(const franka::RobotState& state, std::array<double, 42>& zeroJacobian_array){
  VectorXd Observers::get_O_F_ext_hat_K(const franka::RobotState& state, const franka_hardware::ModelBase* model){
    std::array<double, 42> zeroJacobian_array = model->zeroJacobian(franka::Frame::kStiffness, state);
    // std::array<double, 42> zeroJacobian_array = model.zeroJacobian(franka::Frame::kStiffness, state);
    
    // different to the franka O_F_ext_hat_K!! mario: wrench acting on the stiffness frame, expressed in the base frame! 
    // franka: wrench acting on the stiffness frame transfered to the base frame(forces times lever arm O_T_K), expressed in the base frame
    // Discussed with Hamid; seems like it is a bug with franka, at least libfranka 0.9.2
    Map<const Matrix<double, 6, 7>> zeroJacobian(zeroJacobian_array.data());

    VectorXd tau_ext_hat_filtered = get_tau_ext_hat_filtered(state);

    MatrixXd zeroJacobianT = zeroJacobian.transpose();
    MatrixXd pinv_zeroJacobianT = (zeroJacobian*zeroJacobianT).inverse()*zeroJacobian;

    VectorXd O_F_ext_hat_K = pinv_zeroJacobianT * tau_ext_hat_filtered;
    return O_F_ext_hat_K;
  };

  VectorXd Observers::get_K_F_ext_hat_K(const franka::RobotState& state, std::array<double, 42>& k_Jacobian_array){
    // std::array<double, 42> xx_Jacobian_array = model.bodyJacobian(franka::Frame::kStiffness, state);
    VectorXd K_F_ext_hat_K = get_xx_F_ext_hat_K(state, k_Jacobian_array);
    return K_F_ext_hat_K;
  };

  VectorXd Observers::get_xx_F_ext_hat_K(const franka::RobotState& state, std::array<double, 42>& xx_Jacobian_array){ 
    // std::array<double, 42> xx_Jacobian_array = model.bodyJacobian(frame, state);
    Map<const Matrix<double, 6, 7>> xxJacobian(xx_Jacobian_array.data());

    VectorXd tau_ext_hat_filtered = get_tau_ext_hat_filtered(state);

    MatrixXd xxJacobianT = xxJacobian.transpose();
    MatrixXd pinv_xxJacobianT = (xxJacobian*xxJacobianT).inverse()*xxJacobian;

    VectorXd xx_F_ext_hat_K = pinv_xxJacobianT * tau_ext_hat_filtered;
    return xx_F_ext_hat_K;
  };
  
  // Not using it
  // double Observers::get_obj_mass(const franka::RobotState& state, std::array<double, 42>& zeroJacobian_array){
  //   Vector3d O_gravity = _dyn->get_O_gravity();
  //   // VectorXd O_F_ext_hat_K_head = get_O_F_ext_hat_K(state, zeroJacobian_array).head(3)*-1.0;
  //   VectorXd O_F_ext_hat_K_head = get_K_F_ext_hat_K(state, zeroJacobian_array).head(3)*-1.0;

  //   VectorXd O_gravity_norm = O_gravity.normalized();
  //   double Wz_F_ext_hat_K =  O_gravity_norm.dot(O_F_ext_hat_K_head);
  //   double mass = Wz_F_ext_hat_K/O_gravity.norm();

  //   // std::cout << "wz:" << Wz_F_ext_hat_K << std::endl;
  //   // std::cout << "Fext norm:" << O_F_ext_hat_K_head.transpose() << std::endl;
  //   return mass;
  // };

  // Not using it
  // VectorXd Observers::get_K_com_ext_K(const franka::RobotState& state, const franka::Model& model, VectorXd O_gravity){
  //   VectorXd K_F_ext_hat_K = get_K_F_ext_hat_K(state, model);
  //   // TODO: clarify with Mario: get_obj_mass or get_ext_mass?
  //   double mass = get_ext_mass(state, model, O_gravity);

  //   VectorXd K_com_ext_K = (-1.0)*K_F_ext_hat_K.tail(3)/(mass*O_gravity.norm());
  //   return K_com_ext_K;
  // };
} // namespace panda_identification