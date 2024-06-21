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
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include <adaptnotch/adaptnotch.h>

#include "SnapdragonImuManager.hpp"
#include "SnapdragonControllerManager.hpp"
#include "SnapdragonEscManager.hpp"

#ifndef STRUCT_
#define STRUCT_
#include "structs.h"
#endif

namespace Snapdragon {
  class ObserverManager;
}


class Snapdragon::ObserverManager : public Snapdragon::Imu_IEventListener {
public:

  // Filter gains struct
  typedef struct {
    float Kp;  // Position Kalman gain
    float Kv;  // Velocity Kalman gain
    float Kq;  // Quaterion complimentary filter gain
    float Kab; // Accel bias Kalman gain
    float Kgb; // Gyro bias Kalman gain
    double fc_acc_xy, fc_acc_z; // cut-off freq for accelerometer LPF
    double fc_gyr; // cut-off freq for gyro LPF
    bool sfpro; // sfpro 820 (8096) has a different IMU to body transform (for SPI 1 IMU)

    bool anotch_enable; ///< should we use the adaptive gyro notch filter?
    adaptnotch::AdaptiveNotch::Params anotch_params; ///< adaptive notch parameters
  } InitParams;

  // IMU data struct
  typedef struct 
  {
    float lin_accel[3], ang_vel[3];
    uint32_t sequence_number;
    uint64_t current_timestamp_ns;
  } Data;

  // Filter state
  struct State
  {
    uint32_t sequence_number;
    uint64_t current_timestamp_ns;
    Vector pos;        // [x y z]
    Vector vel;        // [vx vy vz]
    Vector w;          // [x y z]
    Vector accel_bias; // [x y z]
    Vector gyro_bias;  // [x y z]
    Quaternion q;      // [w x y z]

    void updateState(Vector pu, Quaternion qu){
      pos.x = pu.x; pos.y = pu.y; pos.z = pu.z;
      q.w = qu.w; q.x = qu.x; q.y = qu.y; q.z = qu.z;
    }
  } ;

  /**
   * Constructor
   **/
  ObserverManager();

  /**
   * Initalizes the Observer Manager with filter and controller parameters
   * @param params
   *  The structure that holds the filter parameters.
   * @param smcparams
   *  The structure that holds the attitude controller parameters.
   * @return 
   *  0 = success
   * otherwise = failure.
   **/
  int32_t Initialize
  (
    const std::string& vehname,
    const Snapdragon::ObserverManager::InitParams& params,
    const Snapdragon::ControllerManager::InitParams& smcparams
  );


  /**
   * Start the IMU, attitude controller, comm manager.
   * @return 
   *   0 = success
   *  otherwise = failure;
   **/
  int32_t Start();

  /**
   * Stop the IMU, attitude controller, comm manager.
   * @return 
   *   0 = success;
   * otherwise = failure.
   **/
  int32_t Stop();

  /**
   * The IMU callback handler to process the accel/gyro data.
   * @param samples
   *  The IMU samples to be processed.
   * @param count
   *  The number of samples in the buffer.
   * @return int32_t
   *  0 = success;
   * otherwise = false;
   **/
  int32_t Imu_IEventListener_ProcessSamples( sensor_imu* samples, uint32_t count );


  /**
   * Propagate IMU data to get vehicle's full state.
   * @param Filter state (modified by ref)
   * @param IMU data
   * @param Time difference between IMU measurements
   * @return 
   *   0 = success
   *  otherwise = failure;
   **/
  int32_t propagateState( Snapdragon::ObserverManager::State& state, Snapdragon::ObserverManager::Data data, float dt );

  /** 
   * Update filter state.
   * @param Filter state (modified by ref)
   * @param Position update
   * @param Quaternion update
   * @param timestamp_us external pose measurement timestamp in microseconds
   * @return 
   *   0 = success
   *  otherwise = failure;
   **/
  int32_t updateState(Snapdragon::ObserverManager::State& state, Vector pos, Quaternion q, uint64_t timestamp_us);

  /**
   * Update desired state of attitude controller.
   * @param Desired attitude and angular rates
   * @return 
   *   0 = success
   *  otherwise = failure;
   **/
  int32_t updateSMCState ( desiredAttState newDesState );

  /**
   * @brief      Filters the new IMU samples
   *
   * @param      imu    Current IMU state to be updated
   * @param[in]  accel  New accelerometer samples
   * @param[in]  gyro   New rate gyro samples
   */
  void filter(Data& imu, const float accel[3], const float gyro[3]);

  Snapdragon::EscManager * const escManager() const { return esc_man_ptr_; }

  bool isCalibrated() const { return calibrated_; }

  /**
   * Destructor
   **/
  virtual ~ObserverManager();

  Snapdragon::ObserverManager::Data           imu_data_;
  Snapdragon::ObserverManager::State          state_;
  Snapdragon::ControllerManager::motorThrottles  smc_motors_;
  Snapdragon::ControllerManager::controlData  smc_data_;
  std::atomic<bool>                           calibrated_;

private:
  std::atomic<bool> initialized_, got_pose_;
  Snapdragon::ObserverManager::InitParams observer_params_;
  Snapdragon::ImuManager*            imu_man_ptr_;
  Snapdragon::ControllerManager*     smc_man_ptr_;
  Snapdragon::EscManager*            esc_man_ptr_;
  std::mutex                         sync_mutex_;
  uint64_t last_pose_update_us_; ///< timestamp of last external pose measurement

  double alpha_accel_xy_, alpha_accel_z_, alpha_gyro_; ///< RC LPF gains

  using AdaptiveNotchPtr = std::unique_ptr<adaptnotch::AdaptiveNotch>;
  std::array<AdaptiveNotchPtr, 3> anotch_gyr_; ///< adaptive notch filters for gyro

  int32_t CleanUp();

  /**
   * @brief      Compute the LPF gain (i.e., alpha)
   *
   * @param[in]  fc    cut-off frequency
   * @param[in]  dT    sample period
   *
   * @return     LPF gain for y = (1-alpha) * y + alpha * x
   */
  inline double computeLPFGain(double fc, double dT) const
  {
    const double RC = 1. / (2 * M_PI * fc);
    return dT / (RC + dT);
  }
};
