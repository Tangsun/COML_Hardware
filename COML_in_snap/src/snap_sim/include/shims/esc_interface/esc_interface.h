/**
 * @file esc_interface.h
 * @brief ESC Interface API
 * @author Parker Lusk <plusk@mit.edu>
 * @date 27 June 2019
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "esc_interface/esc_datatypes.h"
#include "server.h"

namespace acl {

  class ESCInterface
  {
  public:
    ESCInterface(const std::string& vehname, uint8_t num_pwm)
        : vehname_(vehname), num_pwm_(num_pwm) {};
    ~ESCInterface();

    bool init();
    void close();

    bool arm();
    bool disarm();
    bool is_armed() const { return armed_; }
    bool hw_error() const { return hw_error_; }

    /**
     * @brief      Write out new PWM values
     *
     * @param[in]  pwm   array of pwm values
     * @param[in]  len   the array length
     *
     * @return     true if update was successful (i.e., hardware responds)
     */
    bool update(const uint16_t * pwm, uint16_t len);

    static constexpr uint16_t PWM_MIN_PULSE_WIDTH = 1000;
    // TODO: calibrate ESCs and use the full range
    static constexpr uint16_t PWM_MAX_PULSE_WIDTH = 1800;

  private:
    std::string vehname_; ///< vehicle name
    uint8_t num_pwm_ = 0; ///< how many pwms do we expect to control?
    bool initialized_ = false; ///< pwm peripheral initialized (needs closing)
    bool armed_ = false; ///< software arming allows writing non PWM_MIN values
    bool hw_error_ = false; ///< cannot communicate with pwm peripheral/device

    // IPC server to send esc commands
    std::unique_ptr<acl::ipc::Server<esc_commands>> escserver_;

    /**
     * @brief      Commands all PWMs to PWM_MIN_PULSE_WIDTH
     *
     * @return     true if PWM update was successful
     */
    bool min_throttle();

    /**
     * @brief      Sets the PWM unconditionally. Necessary to set
     *             min_throttle without checking arm status
     *
     * @param[in]  pwm   The array of values to set
     * @param[in]  len   The length of the array
     *
     * @return     True if set
     */
    bool set_pwm(const uint16_t * pwm, uint16_t len);
  };

} // ns acl
