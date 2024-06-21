/**
 * @file outer_loop.h
 * @brief Outer loop trajectory tracking snap-stack controller
 * @author Parker Lusk <parkerclusk@gmail.com>
 * @author Jesus Tordesillas Torres <jotrde@mit.edu>
 * @date 20 August 2020
 *
 * @note We use ROS standard frames: inertial ENU with body flu
 */

#include <cmath>
#include <vector>

#include <Eigen/Dense>

namespace acl {
namespace outer_loop {

  /**
   * @brief      Current vehicle state
   */
  struct State
  {
    double t = -1; ///< time associated with this state, [sec]
    Eigen::Vector3d p; ///< position w.r.t world frame
    Eigen::Vector3d v; ///< velocity w.r.t world frame
    Eigen::Quaterniond q; ///< orientation of body w.r.t world frame
    Eigen::Vector3d w; ///< angular vel of body w.r.t world expressed in body

    State()
    : p(Eigen::Vector3d::Zero()), v(Eigen::Vector3d::Zero()),
      q(Eigen::Quaterniond::Identity()), w(Eigen::Vector3d::Zero())
    {}
  };

  /**
   * @brief      Trajectory tracking goal from a high-level planner
   */
  struct Goal
  {
    enum class Mode { POS_CTRL, VEL_CTRL, ACC_CTRL };
    Mode mode_xy = Mode::POS_CTRL; ///< determines which error signals to use
    Mode mode_z = Mode::POS_CTRL; ///< determines which error signals to use

    double t = -1; ///< time associated with this goal, [sec]
    Eigen::Vector3d p; ///< position w.r.t world frame
    Eigen::Vector3d v; ///< velocity w.r.t world frame
    Eigen::Vector3d a; ///< accel    w.r.t world frame
    Eigen::Vector3d j; ///< jerk     w.r.t world frame

    double psi; ///< angle as defined in Sec. III of https://arxiv.org/pdf/2103.06372.pdf  See also https://link.springer.com/chapter/10.1007/978-3-030-28619-4_20
    double dpsi; ///< d{psi}/dt

    Goal()
    : p(Eigen::Vector3d::Zero()), v(Eigen::Vector3d::Zero()),
      a(Eigen::Vector3d::Zero()), j(Eigen::Vector3d::Zero()),
      psi(0), dpsi(0)
    {}
  };

  /**
   * @brief      Log of all control signals, including intermediate values
   */
  struct ControlLog
  {
    Eigen::Vector3d p;
    Eigen::Vector3d p_ref;
    Eigen::Vector3d p_err;
    Eigen::Vector3d p_err_int;
    Eigen::Vector3d v;
    Eigen::Vector3d v_ref;
    Eigen::Vector3d v_err;
    Eigen::Vector3d a_ff;
    Eigen::Vector3d a_fb;
    Eigen::Vector3d j_ff;
    Eigen::Vector3d j_fb;
    Eigen::Quaterniond q;
    Eigen::Quaterniond q_ref;
    Eigen::Vector3d w;
    Eigen::Vector3d w_ref;
    Eigen::Vector3d F_W; ///< total desired force [N], expr in world frame

    Goal::Mode mode_xy, mode_z; ///< current trajectory tracking mode

    ControlLog()
    : p(Eigen::Vector3d::Zero()), p_ref(Eigen::Vector3d::Zero()),
      p_err(Eigen::Vector3d::Zero()), p_err_int(Eigen::Vector3d::Zero()),
      v(Eigen::Vector3d::Zero()), v_ref(Eigen::Vector3d::Zero()),
      v_err(Eigen::Vector3d::Zero()),
      a_ff(Eigen::Vector3d::Zero()), a_fb(Eigen::Vector3d::Zero()),
      j_ff(Eigen::Vector3d::Zero()), j_fb(Eigen::Vector3d::Zero()),
      q(Eigen::Quaterniond::Identity()), q_ref(Eigen::Quaterniond::Identity()),
      w(Eigen::Vector3d::Zero()), w_ref(Eigen::Vector3d::Zero()),
      F_W(Eigen::Vector3d::Zero()),
      mode_xy(Goal::Mode::POS_CTRL), mode_z(Goal::Mode::POS_CTRL)
    {}
  };

  /**
   * @brief      Attitude-loop command
   */
  struct AttCmd
  {
    Eigen::Quaterniond q; ///< desired attitude
    Eigen::Vector3d w; ///< desired angular rate
    Eigen::Vector3d F_W; ///< total desired force [N], expr in world frame

    AttCmd()
    : q(Eigen::Quaterniond::Identity()), w(Eigen::Vector3d::Zero()),
      F_W(Eigen::Vector3d::Zero())
    {}
  };

  class OuterLoop
  {
  public:

    /**
     * @brief      Outer loop trajectory tracking paramters
     */
    struct Parameters
    {
      double mass; ///< vehicle mass, [kg]
      Eigen::Vector3d Kp, Ki, Kd; ///< x-y-z PID trajectory tracking gains
      Eigen::Vector3d maxPosErr; ///< position err [m] above this is clamped
      Eigen::Vector3d maxVelErr; ///< velocity err [m/s] above this is clamped
    };

  public:
    OuterLoop(const Parameters& params);
    ~OuterLoop() = default;

    /**
     * @brief      Resets any internal state for a new flight
     */
    void reset();

    /**
     * @brief      Given a goal (p,v,a,j) and the current state, produce a
     *             reference attitude and attitude rate command.
     *
     * @param[in]  t      Current time, in seconds
     * @param[in]  state  Current state
     * @param[in]  goal   Desired pos, vel, acc, jrk
     *
     * @return     An attitude command (q, omega)
     */
    AttCmd computeAttitudeCommand(double t, const State& state, const Goal& goal);

    /**
     * @brief      Resets the log and sets the state to the current state.
     *             This is useful since computeAttitudeCommand may not always
     *             be called. Since this resets the log, it should be called
     *             *before* computeAttitudeCommand.
     *
     * @param[in]  state  Current state
     */
    void updateLog(const State& state);
    const ControlLog& getLog() const { return log_; }

    /**
     * @brief      Set/update the parameters at runtime for the PID controller
     *             on the x-y axes in the Inertial frame.
     *
     * @param[in]  kp    Proportional gain.
     * @param[in]  ki    Integral gain.
     * @param[in]  kd    Derivative gain.  
     *
     */
    void setHorizontalPositionPidParams(const double kp, const double ki, const double kd);

    /**
     * @brief      Set/update the parameters at runtime for the PID controller
     *             on the z axis in the Inertial frame.
     *
     * @param[in]  kp    Proportional gain.
     * @param[in]  ki    Integral gain.
     * @param[in]  kd    Derivative gain.  
     *
     */
    void setAltitudePidParams(const double kp, const double ki, const double kd);

  private:
    const Eigen::Vector3d GRAVITY = (Eigen::Vector3d() << 0, 0, -9.80665).finished();
    Parameters params_;
    ControlLog log_;
    Goal::Mode mode_xy_last_, mode_z_last_; ///< used to detect mode change
    Eigen::Vector3d a_fb_last_; ///< last accel feedback value
    Eigen::Vector3d j_fb_last_; ///< last jerk feedback value
    double t_last_; ///< time [sec] of last control loop

    /**
     * @brief      Accumulator for I term
     */
    class Integrator
    {
    public:
      Integrator() = default;
      ~Integrator() = default;
      // TODO: anti-windup
      void increment(double inc, double dt) { value_ += inc * dt; }
      void reset() { value_ = 0; }
      double value() const { return value_; }
    private:
      double value_ = 0;
    };

    Integrator Ix_, Iy_, Iz_; ///< integrators to accumulate pos error for PID

    /**
     * @brief      Uses PID to drive position and velocity error to zero by
     *             producing an acceleration feedback signal. This is added
     *             to any feedforward acceleration goal to yield the required
     *             force vector, expressed in world frame.
     *
     * @param[in]  dt     Timestep [sec]
     * @param[in]  state  Current vehicle state
     * @param[in]  goal   Goal pos, vel, acc, jrk
     *
     * @return     The world-frame force required to achieve/maintain goal.
     */
    Eigen::Vector3d getForce(double dt, const State& state, const Goal& goal);

    /**
     * @brief      Computes the attitude required for the vehicle to produce
     *             the desired world force. This is done by finding the minimum
     *             rotation required from the +z unit vector to the force vec.
     *
     * @param[in]  state  Current vehicle state
     * @param[in]  goal   Goal
     * @param[in]  F_W    Previously computed force vector, expr in world frame
     *
     * @return     Desired rotation of body w.r.t world
     */
    Eigen::Quaterniond getAttitude(const State& state, const Goal& goal,
                                    const Eigen::Vector3d& F_W);

    /**
     * @brief      Computes rate command that is dynamically consistent with
     *             the way the desired attitude is found.
     *
     * @param[in]  dt     Timestep [sec]
     * @param[in]  state  Current vehicle state
     * @param[in]  goal   Goal (used for desired heading rate)
     * @param[in]  F_W    Previously computed force vector, expr in world frame
     * @param[in]  a_fb   Acceleration feedback from getForce PID
     * @param[in]  q_ref  Previously computed desired rot of body w.r.t world
     *
     * @return     Desired body rates w.r.t world, expr in body
     */
    Eigen::Vector3d getRates(double dt, const State& state,
                                  const Goal& goal, const Eigen::Vector3d& F_W,
                                  const Eigen::Vector3d& a_fb,
                                  const Eigen::Quaterniond& q_ref);
  };

} // ns outer_loop
} // ns acl
