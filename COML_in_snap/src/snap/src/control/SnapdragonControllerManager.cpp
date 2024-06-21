/****************************************************************************
 *   Copyright (c) 2017 Brett T. Lopez. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name snap nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/

#include "SnapdragonControllerManager.hpp"
#include "SnapdragonUtils.hpp"
#include "SnapdragonDebugPrint.h"
#include <iostream>

Snapdragon::ControllerManager::ControllerManager() {
	initialized_ = false;
}

Snapdragon::ControllerManager::~ControllerManager() {
	// TODO: add destructor
}

void Snapdragon::ControllerManager::Initialize(const Snapdragon::ControllerManager::InitParams& smc_params) {
	smc_params_ = smc_params;
	smc_state_ = attState();
  smc_des_ = desiredAttState();
	initialized_ = true;

	Omegad_dot_last_ = Eigen::Vector3d::Zero();
	Omegad_last_ = Eigen::Vector3d::Zero();

	if (smc_params_.mixer == Mixer::QUAD_X) {

		f2wrench_ = Eigen::Matrix<double,4,4>::Zero();

    // Length projected. We assume symmetry here (motors placed every 90 deg)
    // and first motor placed +45 degrees wrt body axes x
		const double lp = smc_params_.l * std::cos(M_PI / 4.0);

    f2wrench_ << 1.,   1.,   1.,    1.,
                 lp,   lp,  -lp,   -lp,
                -lp,   lp,   lp,   -lp,
                 lp,  -lp,   lp,   -lp;

		wrench2f_ = f2wrench_.inverse();

	} else if (smc_params_.mixer == Mixer::HEX) {

		f2wrench_= Eigen::Matrix<double,4,6>::Zero();

		// Assuming symmetry here (i.e. motors placed every 60 degrees),
		// and first motor placed +30 degrees wrt body axes x.
		// See photo https://gitlab.com/mit-acl/fsw/vehicle-builds/hx/-/wikis/home

    // Angle between two motors.
    // Note that a/2 is the angle between first motor and body axes x.
		const double a = 2 * (M_PI / 6.);

		const double l = smc_params_.l;
		const double lsa2 = l * std::sin(a / 2.);
		const double lca2 = l * std::cos(a / 2.);
		const double cd = smc_params_.cd;
		const double rx = smc_params_.com.at(0);
		const double ry = smc_params_.com.at(1);

    f2wrench_ << 1.,    1.,   1.,     1.,    1.,   1.,
                 lsa2 - ry,  l - ry,    lsa2 - ry,  -lsa2 - ry, -l - ry,   -lsa2 - ry,
                -lca2 + rx,  0. + rx,   lca2 + rx,   lca2 + rx,  0. + rx,  -lca2 + rx,
                 cd,   -cd,   cd,    -cd,    cd,  -cd;

		// It´s an undertermined system (i.e., many solutions). Let us choose the option that minimizes ||f||,
    // see last equation of http://people.csail.mit.edu/bkph/articles/Pseudo_Inverse.pdf
    // TODO: instead of solving min-norm soln manually, should used SVD for numerics
    wrench2f_ = f2wrench_.transpose() * (f2wrench_ * f2wrench_.transpose()).inverse();

	} else { // invalid
		f2wrench_= Eigen::Matrix<double,8,8>::Zero();
		wrench2f_= Eigen::Matrix<double,8,8>::Zero();
	}
}

void Snapdragon::ControllerManager::updateDesiredAttState(desiredAttState &desState, desiredAttState newdesState) {
	// Lock thread
  	std::lock_guard<std::mutex> lock( sync_mutex_ );
  	desState = newdesState;
}

void Snapdragon::ControllerManager::updateAttState(attState &attState, Quaternion q, Vector w) {
	// Lock thread
  	std::lock_guard<std::mutex> lock( sync_mutex_ );
  	attState.q = q;
  	attState.w = w;
}

void Snapdragon::ControllerManager::updateMotorCommands(double dt, Snapdragon::ControllerManager::motorThrottles &throttles, desiredAttState desState, attState attState ) {
	if (desState.power && initialized_) {
		// Lock thread
  	std::lock_guard<std::mutex> lock( sync_mutex_ );

  	// For referenced equations, see "Geometric tracking control of a quadrotor UAV on SE(3)"
    // https://ieeexplore.ieee.org/document/5717652

  	//get rotations and omegas (both current and desired)
  	const Eigen::Matrix3d R = Eigen::Quaterniond(attState.q.w, attState.q.x, attState.q.y, attState.q.z).toRotationMatrix();
  	const Eigen::Matrix3d Rd = Eigen::Quaterniond(desState.q.w, desState.q.x, desState.q.y, desState.q.z).toRotationMatrix();
  	const Eigen::Vector3d Omega = Eigen::Vector3d(attState.w.x, attState.w.y, attState.w.z); // current angular velocity of the UAV's c.o.m. in the BODY frame
  	const Eigen::Vector3d Omegad = Eigen::Vector3d::Zero(); //Eigen::Vector3d(desState.w.x, desState.w.y, desState.w.z); // desired angular velocity of the UAV's c.o.m. in the BODY frame

    // numerically differentiate. TODO: could be avoided if we had snap
    Eigen::Vector3d Omegad_dot = (Omegad - Omegad_last_) / dt;

    // low-pass filter differentiation with time constant tau [sec]
    static constexpr double tau = 0.1;
    const double alpha = dt / (tau + dt);
    Omegad_dot = alpha * Omegad_dot + (1 - alpha) * Omegad_dot_last_;

		// save for next time
		Omegad_dot_last_ = Omegad_dot;
		Omegad_last_ = Omegad;

	Omegad_dot = Eigen::Vector3d::Zero();

  	// compute error in rotation
  	const Eigen::Vector3d eR = 0.5 * Vee(Rd.transpose() * R - R.transpose() * Rd); // Eq. 10

  	// compute error in omega
  	const Eigen::Vector3d eOmega = (Omega - R.transpose() * Rd * Omegad); // Eq. 11

  	// Compute moment [Nm]
  	const Eigen::Vector3d M = -smc_params_.Kr.cwiseProduct(eR)
                            - smc_params_.Komega.cwiseProduct(eOmega)
                            + Omega.cross(smc_params_.J * Omega)
  						              - smc_params_.J*(Hat(Omega) * R.transpose() * Rd * Omegad
                            - R.transpose()*Rd*Omegad_dot); // Eq. 16

  	// Compute total thrust [N]
  	const double T = Eigen::Vector3d(desState.F_W.x, desState.F_W.y, desState.F_W.z).dot(R*Eigen::Vector3d::UnitZ()); // Eq. 15
   
    // Apply mapping to [T;M] and obtain f (in N) for each of the motors
		Eigen::VectorXd f = wrench2f_ * Eigen::Vector4d(T, M[0], M[1], M[2]);

		// Apply curve to f [N] to obtain throttle (\in [0,1]) for each of the motors, and saturate it
		// TODO: one possible improvement is to saturate it such that Mx, My have the highest priority, then T, and then Mz. 
		// See https://www.ifi.uzh.ch/dam/jcr:5f3668fe-1d4e-4c2b-a190-8f5608f40cf3/RAL17_Faessler.pdf
		for(size_t i=0; i<f.size(); i++) {
      		throttles.throttle[i] = saturate(f2Throttle(f[i], i), 0., 1.);
		}

		// The rest of the f will be zero
		for(size_t i=f.size(); i<=7; i++){
			throttles.throttle[i] = 0.0;
		}

		// Update smc_data struct
		smc_data_.q_des = desState.q;
		smc_data_.q_act = attState.q;
		Quaternion qe; //only used for logging
		qConjProd(qe,attState.q,desState.q);
		smc_data_.q_err = qe;

		smc_data_.w_des = desState.w;
		smc_data_.w_act = attState.w;
		smc_data_.w_err.x = eOmega[0]; smc_data_.w_err.y = eOmega[1]; smc_data_.w_err.z = eOmega[2];
	} else {
		throttles.setAllZero();
	}

}


// converts f (in N) to throttle
double Snapdragon::ControllerManager::f2Throttle(double thrust_per_motor, size_t motor_id)
{
  // Use a polynomial model to map thrust_per_motor to throttle_per_motor.
  double throttle_per_motor  = 0.0;
  if (smc_params_.motor_spin.at(motor_id) < 0){
    // Clockwise spinning rotor
	throttle_per_motor = evalPoly(smc_params_.polyF_cw, thrust_per_motor);
  }else{
	// Counter-clockwise spinning rotor
	throttle_per_motor = evalPoly(smc_params_.polyF_ccw, thrust_per_motor);
  } 
  return throttle_per_motor;
}


double Snapdragon::ControllerManager::evalPoly(const std::vector<double>& coeffs, double x) const
{
  // assumption: coeffs is ordered as [a_n ... a_0] s.t y = a_n*x^n + ... + a_0
  double y = 0.0;
  for (size_t i=0; i<coeffs.size(); ++i) {
    y += coeffs[i] * std::pow(x, coeffs.size() - 1 - i);
  }
  return y;
}
