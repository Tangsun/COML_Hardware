/**
 * @file sensor_imu_api.h
 * @brief Shim for simulated Snapdragon IMU API
 * @author Parker Lusk <parkerclusk@gmail.com>
 * @date 19 December 2019
 */

#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "sensor-imu/sensor_datatypes.h"

#define SENSOR_IMU_AND_ATTITUDE_API_VERSION "sim-imu"

// Dummy handle to represent the implementation of the sensor_imu_attitude_api
typedef void sensor_handle;

// Returns pointer to new int; just for compatibility
sensor_handle* sensor_imu_attitude_api_get_instance(const std::string& vehname);

// Returns SENSOR_IMU_AND_ATTITUDE_API_VERSION for compatilibity
char* sensor_imu_attitude_api_get_version(sensor_handle* handle);

// Initialize the simulated IMU, return 0 on success
int16_t sensor_imu_attitude_api_initialize(sensor_handle* handle, uint8_t unused);

// Shutdown and clean up the simulated IMU, return 0 on success
int16_t sensor_imu_attitude_api_terminate(sensor_handle* handle);

// Gets raw imu data, return 0 on success
int16_t sensor_imu_attitude_api_get_imu_raw(sensor_handle* handle,
                                            sensor_imu* dataArray,
                                            int32_t max_count,
                                            int32_t* returned_sample_count);

// Makes sure simulated IMU is initialized, for compatibility, return 0 on success
int16_t sensor_imu_attitude_api_wait_on_driver_init(sensor_handle* handle);
