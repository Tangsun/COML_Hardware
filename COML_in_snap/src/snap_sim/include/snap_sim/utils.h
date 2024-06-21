/*!
 * \file utils.hpp
 *
 * Simple math utility functions for the ACL
 *
 *  Created on: Mar 15, 2013
 *      Author: Mark Cutler
 *     Contact: markjcutler@gmail.com
 *
 *  Parker Lusk <parkerclusk@gmail.com>
 *  9 Dec 2019
 *
 */

#pragma once

#include <cmath>
#include <functional>
#include <valarray>

namespace acl {

/**
 * @brief      Clamp function to saturate between a low and high value
 *
 * @param[in]  val      The value to clamp
 * @param[in]  lower    The lower bound
 * @param[in]  upper    The upper bound
 * @param      clamped  Wether or not the value was clamped
 *
 * @tparam     T        The template type
 *
 * @return     The clamped value
 */
template<typename T>
static T clamp(const T& val, const T& lower, const T& upper, bool& clamped) {
  if (val < lower) {
    clamped = true;
    return lower;
  }

  if (val > upper) {
    clamped = true;
    return upper;
  }

  clamped = false;
  return val;
}

// ----------------------------------------------------------------------------

/**
 * @brief      Specialization of clamp without clamp indicator
 */
template<typename T>
static T clamp(const T& val, const T& lower, const T& upper) {
  bool clamped = false;
  return clamp(val, lower, upper, clamped);
}

// ----------------------------------------------------------------------------

/**
 * @brief      Saturate the current value of a signal by limit its
 *             rate of change, given its current (desired) value, 
 *             last value, and the timestep.
 *
 * @param[in]  dt        Timestep between current and last value
 * @param[in]  lRateLim  Lower rate limit
 * @param[in]  uRateLim  Upper rate limit
 * @param[in]  v0        The last value
 * @param      v1        The current (des) value, to be rate limited (output)
 */
template<typename T>
static void rateLimit(double dt, const T lRateLim, const T uRateLim,
                      const T v0, T& v1)
{
  // calculate the highest / lowest the components are allowed to attain
  const T upper = v0 + uRateLim * dt;
  const T lower = v0 + lRateLim * dt;

  // make sure current value does not exceed rate of change constraints
  if (v1 > upper) v1 = upper;
  if (v1 < lower) v1 = lower;
}

// ----------------------------------------------------------------------------

/**
 * @brief      Wrap angle so that it is in [-pi, pi]
 *
 * @param[in]  angle  The angle to wrap
 *
 * @return     The wrapped angle
 */
static double wrapToPi(double angle)
{
  if (angle >  M_PI) return angle - 2*M_PI;
  if (angle < -M_PI) return angle + 2*M_PI;
  return angle;
}

// ----------------------------------------------------------------------------

/**
 * @brief      Wrap angle so that it is in [0, 2*pi]
 *
 * @param[in]  angle  The angle to wrap
 *
 * @return     The wrapped angle
 */
static double wrapTo2Pi(double angle)
{
  if (angle > 2*M_PI) return angle - 2*M_PI;
  if (angle < 0)      return angle + 2*M_PI;
  return angle;
}

} // ns acl
