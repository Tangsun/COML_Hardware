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
#include "SnapdragonObserverManager.hpp"
#include "SnapdragonUtils.hpp"
#include "SnapdragonDebugPrint.h"

Snapdragon::ObserverManager::ObserverManager()
{
  imu_man_ptr_ = nullptr;
  smc_man_ptr_ = nullptr;
  esc_man_ptr_ = nullptr;
  initialized_ = false;
  calibrated_ = false;
  got_pose_ = false;
}

Snapdragon::ObserverManager::~ObserverManager()
{
  CleanUp();
}

int32_t Snapdragon::ObserverManager::CleanUp()
{
  // stop the imu manager
  if (imu_man_ptr_ != nullptr)
  {
    imu_man_ptr_->RemoveHandler(this);
    imu_man_ptr_->Terminate();
    delete imu_man_ptr_;
    imu_man_ptr_ = nullptr;
  }
  // stop the attitude control manager
  if (smc_man_ptr_ != nullptr)
  {
    delete smc_man_ptr_;
    smc_man_ptr_ = nullptr;
  }
  // stop the esc manager
  if (esc_man_ptr_ != nullptr)
  {
    esc_man_ptr_->Terminate();
    delete esc_man_ptr_;
    esc_man_ptr_ = nullptr;
  }

  return 0;
}

int32_t Snapdragon::ObserverManager::Initialize(const std::string& vehname,
                                                const Snapdragon::ObserverManager::InitParams& observer_params,
                                                const Snapdragon::ControllerManager::InitParams& smcparams)
{
  state_ = State();
  imu_data_ = Data();

  observer_params_ = observer_params;
  int32_t rc = 0;

  if (rc == 0)
  {
    // initialize the Imu Manager.
    imu_man_ptr_ = new Snapdragon::ImuManager(vehname);
    if (imu_man_ptr_ != nullptr)
    {
      rc = imu_man_ptr_->Initialize();
      imu_man_ptr_->AddHandler(this);
    }
    else
    {
      rc = -1;
    }
    // initialize the Controller Manager
    smc_man_ptr_ = new Snapdragon::ControllerManager();
    smc_man_ptr_->Initialize(smcparams);

    // initialize the Comm Manager
    esc_man_ptr_ = new Snapdragon::EscManager(vehname);
    rc = esc_man_ptr_->Initialize();
  }

  if (rc != 0)
  {
    ERROR_PRINT("Error initializing the Ukf Manager.");
    CleanUp();
  }
  else
  {
    initialized_ = true;
  }
  return 0;
}

int32_t Snapdragon::ObserverManager::Start()
{
  int32_t rc = 0;
  if (initialized_)
  {
    // start the IMU
    rc = imu_man_ptr_->Start();
  }
  else
  {
    ERROR_PRINT("Calling Start without calling intialize");
    rc = -1;
  }
  return rc;
}

int32_t Snapdragon::ObserverManager::Stop()
{
  CleanUp();
  return 0;
}

int32_t Snapdragon::ObserverManager::Imu_IEventListener_ProcessSamples( sensor_imu* imu_samples, uint32_t sample_count ) {
  static constexpr size_t CALIB_SAMPLES = 1000;
  static double Ts = 0; // estimated sample period [sec]

  for (int ii = 0; ii < sample_count; ++ii)
  {
    uint64_t current_timestamp_ns = imu_samples[ii].timestamp_in_us * 1000;

    // make sure we didn't skip too big of a gap in IMU measurements
    static uint64_t last_timestamp = current_timestamp_ns;
    float delta = (current_timestamp_ns - last_timestamp) * 1e-6;
    static constexpr float imu_sample_dt_reasonable_threshold_ms = 12.5;
    if (delta > imu_sample_dt_reasonable_threshold_ms)
    {
      WARN_PRINT("IMU sample dt > %f ms -- %f ms", imu_sample_dt_reasonable_threshold_ms, delta);
    }
    last_timestamp = current_timestamp_ns;

    float lin_acc[3], ang_vel[3];
    static constexpr float kNormG = 9.80665f;

    if (observer_params_.sfpro)
    {  // Excelsior 8096
      // sfpro imu has x coming out the right side, y coming out the front, with z up.
      // Convert from sfpro-imu to body-flu
      lin_acc[0] =  imu_samples[ii].linear_acceleration[1] * kNormG;
      lin_acc[1] = -imu_samples[ii].linear_acceleration[0] * kNormG;
      lin_acc[2] =  imu_samples[ii].linear_acceleration[2] * kNormG;
      ang_vel[0] =  imu_samples[ii].angular_velocity[1];
      ang_vel[1] = -imu_samples[ii].angular_velocity[0];
      ang_vel[2] =  imu_samples[ii].angular_velocity[2];
    }
    else
    {  // Eagle 8074
      // Convert from sf-imu (frd) to body (flu)
      lin_acc[0] =  imu_samples[ii].linear_acceleration[0] * kNormG;
      lin_acc[1] = -imu_samples[ii].linear_acceleration[1] * kNormG;
      lin_acc[2] = -imu_samples[ii].linear_acceleration[2] * kNormG;
      ang_vel[0] =  imu_samples[ii].angular_velocity[0];
      ang_vel[1] = -imu_samples[ii].angular_velocity[1];
      ang_vel[2] = -imu_samples[ii].angular_velocity[2];
    }

    static uint32_t sequence_number_last = 0;
    int num_dropped_samples = 0;
    if (sequence_number_last != 0)
    {
      // The diff should be 1, anything greater means we dropped samples
      num_dropped_samples = imu_samples[ii].sequence_number - sequence_number_last - 1;
      if (num_dropped_samples > 0)
      {
        WARN_PRINT("Current IMU sample = %u, last IMU sample = %u", imu_samples[ii].sequence_number,
                   sequence_number_last);
      }
    }
    sequence_number_last = imu_samples[ii].sequence_number;

    static int count = 0;

    if (!calibrated_) {
      // Update accel and gyro bias initialization
      count++;
      state_.accel_bias.x += lin_acc[0];
      state_.accel_bias.y += lin_acc[1];
      state_.accel_bias.z += (lin_acc[2] - kNormG);
      state_.gyro_bias.x += ang_vel[0];
      state_.gyro_bias.y += ang_vel[1];
      state_.gyro_bias.z += ang_vel[2];

      // accumulate time deltas (in seconds)
      Ts += delta * 1e-3;

      if (count == CALIB_SAMPLES) {
        // Average accel and gyro biases
        state_.accel_bias.x /= CALIB_SAMPLES;
        state_.accel_bias.y /= CALIB_SAMPLES;
        state_.accel_bias.z /= CALIB_SAMPLES;
        state_.gyro_bias.x /= CALIB_SAMPLES;
        state_.gyro_bias.y /= CALIB_SAMPLES;
        state_.gyro_bias.z /= CALIB_SAMPLES;

        // capture estimated sample period
        Ts /= CALIB_SAMPLES;

        // compute LPF alpha parameter
        alpha_accel_xy_ = computeLPFGain(observer_params_.fc_acc_xy, Ts);
        alpha_accel_z_ = computeLPFGain(observer_params_.fc_acc_z, Ts);
        alpha_gyro_ = computeLPFGain(observer_params_.fc_gyr, Ts);

        // initialize adaptive notches
        if (observer_params_.anotch_enable) {
          observer_params_.anotch_params.Fs = 1./Ts;
          for (size_t i=0; i<anotch_gyr_.size(); i++) {
            anotch_gyr_[i].reset(new adaptnotch::AdaptiveNotch(observer_params_.anotch_params));
          }
        }

        calibrated_ = true;
        WARN_PRINT("Calibration complete");
      }
    } else {
      // Apply accel and gyro biases to raw IMU measurements
      float temp_acc[3], temp_gyro[3];
      temp_acc[0]  = (lin_acc[0]-state_.accel_bias.x);
      temp_acc[1]  = (lin_acc[1]-state_.accel_bias.y);
      temp_acc[2]  = (lin_acc[2]-state_.accel_bias.z);
      temp_gyro[0] = (ang_vel[0]-state_.gyro_bias.x);
      temp_gyro[1] = (ang_vel[1]-state_.gyro_bias.y);
      temp_gyro[2] = (ang_vel[2]-state_.gyro_bias.z);

      // filter newly sampled raw IMU data
      filter(imu_data_, temp_acc, temp_gyro);

      imu_data_.sequence_number = sequence_number_last;
      imu_data_.current_timestamp_ns = current_timestamp_ns;

      state_.sequence_number = sequence_number_last;
      state_.current_timestamp_ns = current_timestamp_ns;

      // Propagate state
      propagateState(state_, imu_data_, delta * 1e-3);
    }
  }

  if (calibrated_ && got_pose_) {
    // Update attitude state
    smc_man_ptr_->updateAttState(smc_man_ptr_->smc_state_, state_.q, state_.w);
    
    static double last_timestamp_ns = 0.0;
    const double dt=(state_.current_timestamp_ns - last_timestamp_ns)/1e9; //In seconds
    last_timestamp_ns=state_.current_timestamp_ns;
    // Update Motor commands
    smc_man_ptr_->updateMotorCommands(dt, smc_man_ptr_->throttles_, smc_man_ptr_->smc_des_, smc_man_ptr_->smc_state_);

    // write out to PWM ESCs
    esc_man_ptr_->update(smc_man_ptr_->throttles_.throttle);

    // Copy new motor commands to kf public variable
    std::copy(std::begin(smc_man_ptr_->throttles_.throttle), std::end(smc_man_ptr_->throttles_.throttle), std::begin(smc_motors_.throttle));
    smc_data_ = smc_man_ptr_->smc_data_;
  }

  return 0;
}

int32_t Snapdragon::ObserverManager::propagateState(Snapdragon::ObserverManager::State& state,
                                                    Snapdragon::ObserverManager::Data data, float dt)
{
  // Lock thread to prevent state from being accessed by UpdateState
  std::lock_guard<std::mutex> lock(sync_mutex_);

  Quaternion q = state.q;
  Vector world_lin_acc;

  // Transform accel from body to world frame
  world_lin_acc.x = (1 - 2 * (q.y * q.y + q.z * q.z)) * data.lin_accel[0] +
                    2 * (q.y * q.x - q.z * q.w) * data.lin_accel[1] + 2 * (q.x * q.z + q.y * q.w) * data.lin_accel[2];
  world_lin_acc.y = (1 - 2 * (q.x * q.x + q.z * q.z)) * data.lin_accel[1] +
                    2 * (q.y * q.z - q.x * q.w) * data.lin_accel[2] + 2 * (q.x * q.y + q.z * q.w) * data.lin_accel[0];
  world_lin_acc.z = (1 - 2 * (q.x * q.x + q.y * q.y)) * data.lin_accel[2] +
                    2 * (q.x * q.z - q.y * q.w) * data.lin_accel[0] + 2 * (q.y * q.z + q.x * q.w) * data.lin_accel[1];

  // Accel propogation
  state.pos.x += state.vel.x * dt + 0.5 * dt * dt * world_lin_acc.x;
  state.pos.y += state.vel.y * dt + 0.5 * dt * dt * world_lin_acc.y;
  state.pos.z += state.vel.z * dt + 0.5 * dt * dt * (world_lin_acc.z - 9.80665f);

  state.vel.x += world_lin_acc.x * dt;
  state.vel.y += world_lin_acc.y * dt;
  state.vel.z += (world_lin_acc.z - 9.80665f) * dt;

  // Gyro propogation
  state.q.w -= 0.5 * (q.x * data.ang_vel[0] + q.y * data.ang_vel[1] + q.z * data.ang_vel[2]) * dt;
  state.q.x += 0.5 * (q.w * data.ang_vel[0] - q.z * data.ang_vel[1] + q.y * data.ang_vel[2]) * dt;
  state.q.y += 0.5 * (q.z * data.ang_vel[0] + q.w * data.ang_vel[1] - q.x * data.ang_vel[2]) * dt;
  state.q.z += 0.5 * (q.x * data.ang_vel[1] - q.y * data.ang_vel[0] + q.w * data.ang_vel[2]) * dt;

  // Ensure quaterion is properly normalized
  float norm = sqrt(state.q.w * state.q.w + state.q.x * state.q.x + state.q.y * state.q.y + state.q.z * state.q.z);
  state.q.w /= norm;
  state.q.x /= norm;
  state.q.y /= norm;
  state.q.z /= norm;

  state.w.x = data.ang_vel[0];
  state.w.y = data.ang_vel[1];
  state.w.z = data.ang_vel[2];

  return 0;
}

int32_t Snapdragon::ObserverManager::updateState(Snapdragon::ObserverManager::State& state, Vector pos, Quaternion q,
                                                 uint64_t timestamp_us)
{
  // Lock thread to prevent state from being accessed by PropagateState
  std::lock_guard<std::mutex> lock(sync_mutex_);

  if (!got_pose_)
  {
    // on the first external pose message, just override the filter pose
    // with the external pose measurement.
    state.updateState(pos, q);
    got_pose_ = true;
  }
  else
  {
    float dt = (timestamp_us - last_pose_update_us_) * 1e-6;
    if (dt <= 0)
      return -1;

    Quaternion qe, qa, qu;
    qa.w = state.q.w;
    qa.x = state.q.x;
    qa.y = state.q.y;
    qa.z = state.q.z;
    qu.w = q.w;
    qu.x = q.x;
    qu.y = q.y;
    qu.z = q.z;

    // Generate error quaternion
    qConjProd(qe, qa, qu);

    // Update gyro bias
    state.gyro_bias.x -= observer_params_.Kgb * qe.x;
    state.gyro_bias.y -= observer_params_.Kgb * qe.y;
    state.gyro_bias.z -= observer_params_.Kgb * qe.z;

    Vector err;
    err.x = pos.x - state.pos.x;
    err.y = pos.y - state.pos.y;
    err.z = pos.z - state.pos.z;
    Vector berr;

    // TODO: should this be qa or q?
    berr.x = (1 - 2 * (qa.y * qa.y + qa.z * qa.z)) * err.x + 2 * (qa.y * qa.x + qa.z * qa.w) * err.y +
             2 * (qa.x * qa.z - qa.y * qa.w) * err.z;
    berr.y = (1 - 2 * (qa.x * qa.x + qa.z * qa.z)) * err.y + 2 * (qa.y * qa.z + qa.x * qa.w) * err.z +
             2 * (qa.x * qa.y - qa.z * qa.w) * err.x;
    berr.z = (1 - 2 * (qa.x * qa.x + qa.y * qa.y)) * err.z + 2 * (qa.x * qa.z + qa.y * qa.w) * err.x +
             2 * (qa.y * qa.z - qa.x * qa.w) * err.y;

    // Update accel bias
    state.accel_bias.x -= observer_params_.Kab * (berr.x);
    state.accel_bias.y -= observer_params_.Kab * (berr.y);
    state.accel_bias.z -= observer_params_.Kab * (berr.z);

    // Update state
    state.pos.x += observer_params_.Kp * err.x;
    state.pos.y += observer_params_.Kp * err.y;
    state.pos.z += observer_params_.Kp * err.z;

    state.vel.x += observer_params_.Kv * err.x / dt;
    state.vel.y += observer_params_.Kv * err.y / dt;
    state.vel.z += observer_params_.Kv * err.z / dt;

    state.q.w -= observer_params_.Kq * (state.q.w - q.w);
    state.q.x -= observer_params_.Kq * (state.q.x - q.x);
    state.q.y -= observer_params_.Kq * (state.q.y - q.y);
    state.q.z -= observer_params_.Kq * (state.q.z - q.z);
  }

  last_pose_update_us_ = timestamp_us;
  return 0;
}

int32_t Snapdragon::ObserverManager::updateSMCState(desiredAttState newDesState)
{
  // Update desired attitude
  smc_man_ptr_->updateDesiredAttState(smc_man_ptr_->smc_des_, newDesState);

  return 0;
}


void Snapdragon::ObserverManager::filter(Data& imu, const float accel[3], const float gyro[3])
{
  // RC LPF accelerometer
  imu.lin_accel[0] = (1 - alpha_accel_xy_) * imu.lin_accel[0]  +  alpha_accel_xy_ * accel[0];
  imu.lin_accel[1] = (1 - alpha_accel_xy_) * imu.lin_accel[1]  +  alpha_accel_xy_ * accel[1];
  imu.lin_accel[2] = (1 - alpha_accel_z_)  * imu.lin_accel[2]  +  alpha_accel_z_  * accel[2];

  // RC LPF gyro
  imu.ang_vel[0] = (1 - alpha_gyro_) * imu.ang_vel[0]  +  alpha_gyro_ * gyro[0];
  imu.ang_vel[1] = (1 - alpha_gyro_) * imu.ang_vel[1]  +  alpha_gyro_ * gyro[1];
  imu.ang_vel[2] = (1 - alpha_gyro_) * imu.ang_vel[2]  +  alpha_gyro_ * gyro[2];

  // gyro adaptive notch filtering
  if (observer_params_.anotch_enable) {
    for (size_t i=0; i<3; i++) {
      imu.ang_vel[i] = anotch_gyr_[i]->apply(imu.ang_vel[i]);
    }
    // put the estimated peak frequency in a place it doesn't exactly belong...
    smc_man_ptr_->smc_data_.s.x = anotch_gyr_[0]->peakFreq();
    smc_man_ptr_->smc_data_.s.y = anotch_gyr_[1]->peakFreq();
    smc_man_ptr_->smc_data_.s.z = anotch_gyr_[2]->peakFreq();
  }
}
