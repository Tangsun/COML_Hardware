/**
 * @file esc_interface.cpp
 * @brief Shim for simulated Snapdragon ESC API
 * @author Parker Lusk <plusk@mit.edu>
 * @date 27 December 2019
 */

#include "esc_interface/esc_interface.h"

#include "ipc_common.h"

namespace acl {

bool ESCInterface::init()
{  
  // unique key to access the same shmem location
  const size_t esckey = acl::ipc::createKeyFromStr(vehname_, "esc");
  escserver_.reset(new acl::ipc::Server<esc_commands>(esckey));

  // start in (software) disarmed state
  disarm();

  initialized_ = true;
  return true;
}

// ----------------------------------------------------------------------------

ESCInterface::~ESCInterface()
{
  if (initialized_) close();
}

// ----------------------------------------------------------------------------

void ESCInterface::close()
{
  disarm();
  initialized_ = false;
}

// ----------------------------------------------------------------------------

bool ESCInterface::arm()
{
  armed_ = true;

  // start at zero throttle
  min_throttle();

  return true;
}

// ----------------------------------------------------------------------------

bool ESCInterface::disarm()
{
  // maintain zero throttle to each ESC
  min_throttle();

  armed_ = false;

  return true;
}

// ----------------------------------------------------------------------------


bool ESCInterface::update(const uint16_t * pwm, uint16_t len)
{
  // bail if the ESCs are not software armed
  if (!armed_) return false;

  if (len > num_pwm_) {
    // LOG_ERR("ESC update error: requested to update more pwms than available.");
    return false;
  }

  return set_pwm(pwm, len);
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

bool ESCInterface::min_throttle()
{
  // drive pins to 1000 usec, the min throttle PWM value for ESCs
  uint16_t pwm[num_pwm_];
  for (uint8_t i=0; i<num_pwm_; ++i) pwm[i] = PWM_MIN_PULSE_WIDTH;

  return set_pwm(pwm, num_pwm_);
}

// ----------------------------------------------------------------------------

bool ESCInterface::set_pwm(const uint16_t * pwm, uint16_t len)
{
  esc_commands msg;
  std::memcpy(&msg.pwm, pwm, len*sizeof(uint16_t));
  escserver_->send(msg);

  return true;
}

} // ns acl
