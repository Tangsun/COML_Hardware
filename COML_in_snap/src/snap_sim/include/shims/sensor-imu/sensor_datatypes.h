/**
 * @file sensor_datatypes.h
 * @brief Shim for simulated Snapdragon IMU API
 * @author Parker Lusk <parkerclusk@gmail.com>
 * @date 19 December 2019
 */

#pragma once

// dummy define for compatibility during initialization
#define SENSOR_CLOCK_SYNC_TYPE_MONOTONIC 0

typedef struct {
  uint64_t timestamp_in_us;
  uint32_t sequence_number;
  float linear_acceleration[3]; ///< units of [g/s^2]
  float angular_velocity[3]; ///< units of [rad/s]
} __attribute__((packed)) sensor_imu;
