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

#pragma once
#include <atomic>
#include <mutex>
#include <cstdint>
#include <math.h>
#include <Eigen/Dense>
#include <vector>

#ifndef STRUCT_
#define STRUCT_
#include "structs.h"
#endif

namespace Snapdragon {
  class ControllerManager;
}

class Snapdragon::ControllerManager {
public:

	enum class Mixer { INVALID, QUAD_X, HEX };

	typedef struct {

		Eigen::Vector3d Kr = Eigen::Vector3d(0.4, 0.4, 0.4);
		Eigen::Vector3d Komega = Eigen::Vector3d(0.15, 0.15, 0.15);

    double controlDT = 0.002;

    Eigen::Matrix<double,3,3> J = 0.01*Eigen::Matrix<double,3,3>::Identity(); //Inertia matrix in body frame
    double l    = 0.15; //arm length

    std::vector<double> polyF_cw, polyF_ccw;
	std::vector<int> motor_spin;
	std::vector<double> com; // Center of mass offset in body frame. 


    double cd; // Drag coefficient of each propeller. It is such that the
               // moment created by one motor **around the motor axis** is M=cd*f,
               // where f is the thrust produced by that motor. M in N.m, f in N.

    // Parameter to choose between motor allocation on different vehicles
    Mixer mixer = Mixer::INVALID;
	} InitParams;


	typedef struct {
	    float throttle[8] = {0.,0.,0.,0.,0.,0.,0.,0.};

	    void setAllZero(){
			throttle[0] = 0.;
			throttle[1] = 0.;
			throttle[2] = 0.;
			throttle[3] = 0.;
			throttle[4] = 0.;
			throttle[5] = 0.;
			throttle[6] = 0.;
			throttle[7] = 0.;
	    }
	} motorThrottles;

	struct controlData {
	    Quaternion q_des ;
	    Quaternion q_act ;
	    Quaternion q_err ;
	    Vector w_des ;
	    Vector w_act ;
	    Vector w_err ;
	    Vector s ;
	} ;

	/**
	* Constructor
	**/
	ControllerManager();

	/**
	* Initalizes the Controller Manager with Controller Parameters
	* @param params
	*  The structure that holds the Controller parameters.
	* @return 
	*  0 = success
	* otherwise = failure.
	**/
	void Initialize
	( 
	const Snapdragon::ControllerManager::InitParams& params
	);

	void updateDesiredAttState( desiredAttState &desState, desiredAttState newdesState);
	void updateAttState( attState &attState, Quaternion q, Vector w);
	void updateMotorCommands (double dt, Snapdragon::ControllerManager::motorThrottles &throttles, desiredAttState desState, attState attState );

  /**
   * @brief      Uses a polynomial thrust curve to map thrust to throttle
   *             Thrust curve parameters can be found using a dynamometer.
   *             In case no dynamometer, or for quick testing, the thrust
   *             curve can be assumed linear (not a great assumption) and
   *             given the maximum motor thrust the parameter
   *             b = 1 / f_motor_max.
   *
   *             NOTE: The thrust curve is per motor
   *
   * @param[in]  thrust  	Desired thrust [N]
   * @param[in]  motor_id  	Motor_id: (0, num. motors - 1), used with motor_spin
   * 						to identify the spinning direction and apply the 
   * 						clocwise or counter-clocwise thrust-to-normalized PWM 
   * 						curve. 
   *
   * @return     Throttle value, \in [0, 1]
   */
	double f2Throttle(double thrust_per_motor, size_t motor_id);

	/**
   * @brief      Evaluates a polynomial y = a_n*x^n + ... + a_0
   *             with coeffs [a_n ... a_0] at point x.
   *
   * @param[in]  coeffs  Coeffs ordered as [a_n ... a_0]
   * @param[in]  x       The point to evaluate at
   *
   * @return     Scalar value of polynomial
   */
	double evalPoly(const std::vector<double>& coeffs, double x) const;

	/**
	* Destructor
	*/
	virtual ~ControllerManager();

	desiredAttState  smc_des_;
	attState         smc_state_;
	Snapdragon::ControllerManager::motorThrottles  throttles_;
	Snapdragon::ControllerManager::controlData	   smc_data_;

private:
	std::atomic<bool> initialized_;
	std::atomic<bool> arm_;
	Snapdragon::ControllerManager::InitParams smc_params_;
	std::mutex                    sync_mutex_;

	Eigen::MatrixXd wrench2f_; //Matrix such that [f0;...;fn]=wrench2f*[T;M], where T is the total thrust (scalar, in N), and M is the moment (vector 3x1, in Nm) expressed in the body frame
	Eigen::MatrixXd f2wrench_; //Matrix such that [T;M]=f2wrench*[f0;...;fn], where T is the total thrust (scalar, in N), and M is the moment (vector 3x1, in Nm) expressed in the body frame

	Eigen::Vector3d Omegad_dot_last_;
	Eigen::Vector3d Omegad_last_;

    static Eigen::Vector3d Vee(const Eigen::Matrix3d& in)
    {
      Eigen::Vector3d out;
      out << in(2, 1), in(0, 2), in(1, 0);
      return out;
    }


    static Eigen::Matrix3d Hat(const Eigen::Vector3d& in)
    {
      Eigen::Matrix3d out;
      out << 0.0, -in(2), in(1), 
      		 in(2), 0.0, -in(0),
      		 -in(1), in(0), 0.0;
      return out;
    }
};
