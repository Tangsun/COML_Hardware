;; Auto-generated. Do not edit!


(when (boundp 'snapstack_msgs::Goal)
  (if (not (find-package "SNAPSTACK_MSGS"))
    (make-package "SNAPSTACK_MSGS"))
  (shadow 'Goal (find-package "SNAPSTACK_MSGS")))
(unless (find-package "SNAPSTACK_MSGS::GOAL")
  (make-package "SNAPSTACK_MSGS::GOAL"))

(in-package "ROS")
;;//! \htmlinclude Goal.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(intern "*MODE_POSITION_CONTROL*" (find-package "SNAPSTACK_MSGS::GOAL"))
(shadow '*MODE_POSITION_CONTROL* (find-package "SNAPSTACK_MSGS::GOAL"))
(defconstant snapstack_msgs::Goal::*MODE_POSITION_CONTROL* 0)
(intern "*MODE_VELOCITY_CONTROL*" (find-package "SNAPSTACK_MSGS::GOAL"))
(shadow '*MODE_VELOCITY_CONTROL* (find-package "SNAPSTACK_MSGS::GOAL"))
(defconstant snapstack_msgs::Goal::*MODE_VELOCITY_CONTROL* 1)
(intern "*MODE_ACCELERATION_CONTROL*" (find-package "SNAPSTACK_MSGS::GOAL"))
(shadow '*MODE_ACCELERATION_CONTROL* (find-package "SNAPSTACK_MSGS::GOAL"))
(defconstant snapstack_msgs::Goal::*MODE_ACCELERATION_CONTROL* 2)

(defun snapstack_msgs::Goal-to-symbol (const)
  (cond
        ((= const 0) 'snapstack_msgs::Goal::*MODE_POSITION_CONTROL*)
        ((= const 1) 'snapstack_msgs::Goal::*MODE_VELOCITY_CONTROL*)
        ((= const 2) 'snapstack_msgs::Goal::*MODE_ACCELERATION_CONTROL*)
        (t nil)))

(defclass snapstack_msgs::Goal
  :super ros::object
  :slots (_header _p _v _a _j _psi _dpsi _power _mode_xy _mode_z ))

(defmethod snapstack_msgs::Goal
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:p __p) (instance geometry_msgs::Point :init))
    ((:v __v) (instance geometry_msgs::Vector3 :init))
    ((:a __a) (instance geometry_msgs::Vector3 :init))
    ((:j __j) (instance geometry_msgs::Vector3 :init))
    ((:psi __psi) 0.0)
    ((:dpsi __dpsi) 0.0)
    ((:power __power) nil)
    ((:mode_xy __mode_xy) 0)
    ((:mode_z __mode_z) 0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _p __p)
   (setq _v __v)
   (setq _a __a)
   (setq _j __j)
   (setq _psi (float __psi))
   (setq _dpsi (float __dpsi))
   (setq _power __power)
   (setq _mode_xy (round __mode_xy))
   (setq _mode_z (round __mode_z))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:p
   (&rest __p)
   (if (keywordp (car __p))
       (send* _p __p)
     (progn
       (if __p (setq _p (car __p)))
       _p)))
  (:v
   (&rest __v)
   (if (keywordp (car __v))
       (send* _v __v)
     (progn
       (if __v (setq _v (car __v)))
       _v)))
  (:a
   (&rest __a)
   (if (keywordp (car __a))
       (send* _a __a)
     (progn
       (if __a (setq _a (car __a)))
       _a)))
  (:j
   (&rest __j)
   (if (keywordp (car __j))
       (send* _j __j)
     (progn
       (if __j (setq _j (car __j)))
       _j)))
  (:psi
   (&optional __psi)
   (if __psi (setq _psi __psi)) _psi)
  (:dpsi
   (&optional __dpsi)
   (if __dpsi (setq _dpsi __dpsi)) _dpsi)
  (:power
   (&optional (__power :null))
   (if (not (eq __power :null)) (setq _power __power)) _power)
  (:mode_xy
   (&optional __mode_xy)
   (if __mode_xy (setq _mode_xy __mode_xy)) _mode_xy)
  (:mode_z
   (&optional __mode_z)
   (if __mode_z (setq _mode_z __mode_z)) _mode_z)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; geometry_msgs/Point _p
    (send _p :serialization-length)
    ;; geometry_msgs/Vector3 _v
    (send _v :serialization-length)
    ;; geometry_msgs/Vector3 _a
    (send _a :serialization-length)
    ;; geometry_msgs/Vector3 _j
    (send _j :serialization-length)
    ;; float64 _psi
    8
    ;; float64 _dpsi
    8
    ;; bool _power
    1
    ;; uint8 _mode_xy
    1
    ;; uint8 _mode_z
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; geometry_msgs/Point _p
       (send _p :serialize s)
     ;; geometry_msgs/Vector3 _v
       (send _v :serialize s)
     ;; geometry_msgs/Vector3 _a
       (send _a :serialize s)
     ;; geometry_msgs/Vector3 _j
       (send _j :serialize s)
     ;; float64 _psi
       (sys::poke _psi (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _dpsi
       (sys::poke _dpsi (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; bool _power
       (if _power (write-byte -1 s) (write-byte 0 s))
     ;; uint8 _mode_xy
       (write-byte _mode_xy s)
     ;; uint8 _mode_z
       (write-byte _mode_z s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; geometry_msgs/Point _p
     (send _p :deserialize buf ptr-) (incf ptr- (send _p :serialization-length))
   ;; geometry_msgs/Vector3 _v
     (send _v :deserialize buf ptr-) (incf ptr- (send _v :serialization-length))
   ;; geometry_msgs/Vector3 _a
     (send _a :deserialize buf ptr-) (incf ptr- (send _a :serialization-length))
   ;; geometry_msgs/Vector3 _j
     (send _j :deserialize buf ptr-) (incf ptr- (send _j :serialization-length))
   ;; float64 _psi
     (setq _psi (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _dpsi
     (setq _dpsi (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; bool _power
     (setq _power (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; uint8 _mode_xy
     (setq _mode_xy (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _mode_z
     (setq _mode_z (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;;
   self)
  )

(setf (get snapstack_msgs::Goal :md5sum-) "b0f60475b1346f247f102ae07228cff4")
(setf (get snapstack_msgs::Goal :datatype-) "snapstack_msgs/Goal")
(setf (get snapstack_msgs::Goal :definition-)
      "# Use this message to command the outer loop to track
# a trajectory generated from a high-level trajectory planner.

Header header

# Current time-slice of desired trajectory
geometry_msgs/Point p # position
geometry_msgs/Vector3 v # velocity
geometry_msgs/Vector3 a # acceleration
geometry_msgs/Vector3 j # jerk

float64 psi # angle as defined in Sec. III of https://arxiv.org/pdf/2103.06372.pdf  
# In general, it is will not be the rXYZ yaw Euler angle of the UAV (unless the drone is in a hover condition).
# This is due to the fact that q_psi (see link above) already has some rXYZ yaw on it.
# See also the paper https://link.springer.com/chapter/10.1007/978-3-030-28619-4_20
float64 dpsi #d{psi}/dt

bool power # true if motors should be able to spin

# Trajectory tracking mode constants
uint8 MODE_POSITION_CONTROL     = 0
uint8 MODE_VELOCITY_CONTROL     = 1
uint8 MODE_ACCELERATION_CONTROL = 2

# Trajectory tracking mode for x/y and z components.
# The default is POSITION control, which uses position and velocity error
# to calculate the control effort. VELOCITY control only uses vel error.
# ACCELERATION mode does not use tracking error and could be used to provide
# a control signal computed from something other than the default PID cntrl.
uint8 mode_xy
uint8 mode_z

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
float64 z
")



(provide :snapstack_msgs/Goal "b0f60475b1346f247f102ae07228cff4")


