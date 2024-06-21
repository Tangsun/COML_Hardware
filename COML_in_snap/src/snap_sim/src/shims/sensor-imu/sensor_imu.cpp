#include "sensor-imu/sensor_imu_api.h"

#include "client.h"
#include "ipc_common.h"

sensor_handle* sensor_imu_attitude_api_get_instance(const std::string& vehname)
{
  // unique key to access the same shmem location
  const size_t imukey = acl::ipc::createKeyFromStr(vehname, "imu");
  auto handle = new acl::ipc::Client<sensor_imu>(imukey);
  return reinterpret_cast<void *>(handle);
}

// ----------------------------------------------------------------------------

int16_t sensor_imu_attitude_api_terminate(sensor_handle* handle)
{
  acl::ipc::Client<sensor_imu> * client
                = reinterpret_cast<acl::ipc::Client<sensor_imu>*>(handle);
  delete client;
  return 0;
}

// ----------------------------------------------------------------------------

char* sensor_imu_attitude_api_get_version(sensor_handle* handle)
{
  return SENSOR_IMU_AND_ATTITUDE_API_VERSION;
}

// ----------------------------------------------------------------------------

int16_t sensor_imu_attitude_api_initialize(sensor_handle* handle, uint8_t unused)
{
  return 0;
}

// ----------------------------------------------------------------------------

int16_t sensor_imu_attitude_api_get_imu_raw(sensor_handle* handle,
                                            sensor_imu* dataArray,
                                            int32_t max_count,
                                            int32_t* returned_sample_count)
{
  acl::ipc::Client<sensor_imu> * client
                = reinterpret_cast<acl::ipc::Client<sensor_imu>*>(handle);
  if (client->read(dataArray)) {
    *returned_sample_count = 1;
    return 0;
  } else {
    return 1;
  }
}

// ----------------------------------------------------------------------------

int16_t sensor_imu_attitude_api_wait_on_driver_init(sensor_handle* handle)
{
  return 0;
}