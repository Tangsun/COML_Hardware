/**
 * @file ipc_common.h
 * @brief Common implementation for IPC mechanisim
 * @author Parker Lusk <parkerclusk@gmail.com>
 * @date 19 March 2020
 */

#pragma once

#include <functional>
#include <string>

#include <iostream>

namespace acl {
namespace ipc {

/**
 * @brief      Creates a unique IPC key from string.
 *
 * @param[in]  str   The string
 * @param[in]  salt  The salt---e.g., "imu" or "esc"
 *
 * @return     The unique size_t key
 */
inline size_t createKeyFromStr(const std::string& str, const std::string& salt)
{

  /**
   * Characters chosen:
   *
   * SQ01s imu
   *  ^^^  ^
   *
   * NOTE: We assume the vehicle name is at least four characters long.
   *
   * NOTE: For multiple agents, we are assuming that the vehicle name
   * ends with a unique number, followed by a single letter (an 's').
   */

  // "unique" key to access the same shmem location
  // This is certainly a hack and is not necessarily unique. But given only
  // 4 bytes, it should capture a sufficient amount of information.
  const size_t key = (*(str.end()-4)<<24)
                    |(*(str.end()-3)<<16)
                    |(*(str.end()-2)<<8)
                    |(salt.at(0));
  
  std::cout << "key: " << str << " + " << salt << ": " << key << std::endl;

  // this is *more* unique, but std::hash is not guaranteed to give consistent
  // results across architectures, compilers, implementations, etc.
  // const size_t imukey = std::hash<std::string>()(str + salt);

  return key;
}

} // ns ipc
} // ns acl
