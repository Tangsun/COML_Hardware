/**
 * @file esc_datatypes.h
 * @brief Shim for simulated Snapdragon ESC API
 * @author Parker Lusk <parkerclusk@gmail.com>
 * @date 27 December 2019
 */

#pragma once

struct esc_commands {
  uint16_t pwm[8];
};
