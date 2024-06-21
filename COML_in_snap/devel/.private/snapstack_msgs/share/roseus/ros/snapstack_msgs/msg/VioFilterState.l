;; Auto-generated. Do not edit!


(when (boundp 'snapstack_msgs::VioFilterState)
  (if (not (find-package "SNAPSTACK_MSGS"))
    (make-package "SNAPSTACK_MSGS"))
  (shadow 'VioFilterState (find-package "SNAPSTACK_MSGS")))
(unless (find-package "SNAPSTACK_MSGS::VIOFILTERSTATE")
  (make-package "SNAPSTACK_MSGS::VIOFILTERSTATE"))

(in-package "ROS")
;;//! \htmlinclude VioFilterState.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass snapstack_msgs::VioFilterState
  :super ros::object
  :slots (_header _pose _twist _bw _ba _extrinsics _accel_meas _N _error_cov ))

(defmethod snapstack_msgs::VioFilterState
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:pose __pose) (instance geometry_msgs::Pose :init))
    ((:twist __twist) (instance geometry_msgs::Twist :init))
    ((:bw __bw) (instance geometry_msgs::Vector3 :init))
    ((:ba __ba) (instance geometry_msgs::Vector3 :init))
    ((:extrinsics __extrinsics) (instance geometry_msgs::Pose :init))
    ((:accel_meas __accel_meas) (instance geometry_msgs::Vector3 :init))
    ((:N __N) 0)
    ((:error_cov __error_cov) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _header __header)
   (setq _pose __pose)
   (setq _twist __twist)
   (setq _bw __bw)
   (setq _ba __ba)
   (setq _extrinsics __extrinsics)
   (setq _accel_meas __accel_meas)
   (setq _N (round __N))
   (setq _error_cov __error_cov)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:pose
   (&rest __pose)
   (if (keywordp (car __pose))
       (send* _pose __pose)
     (progn
       (if __pose (setq _pose (car __pose)))
       _pose)))
  (:twist
   (&rest __twist)
   (if (keywordp (car __twist))
       (send* _twist __twist)
     (progn
       (if __twist (setq _twist (car __twist)))
       _twist)))
  (:bw
   (&rest __bw)
   (if (keywordp (car __bw))
       (send* _bw __bw)
     (progn
       (if __bw (setq _bw (car __bw)))
       _bw)))
  (:ba
   (&rest __ba)
   (if (keywordp (car __ba))
       (send* _ba __ba)
     (progn
       (if __ba (setq _ba (car __ba)))
       _ba)))
  (:extrinsics
   (&rest __extrinsics)
   (if (keywordp (car __extrinsics))
       (send* _extrinsics __extrinsics)
     (progn
       (if __extrinsics (setq _extrinsics (car __extrinsics)))
       _extrinsics)))
  (:accel_meas
   (&rest __accel_meas)
   (if (keywordp (car __accel_meas))
       (send* _accel_meas __accel_meas)
     (progn
       (if __accel_meas (setq _accel_meas (car __accel_meas)))
       _accel_meas)))
  (:N
   (&optional __N)
   (if __N (setq _N __N)) _N)
  (:error_cov
   (&optional __error_cov)
   (if __error_cov (setq _error_cov __error_cov)) _error_cov)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; geometry_msgs/Pose _pose
    (send _pose :serialization-length)
    ;; geometry_msgs/Twist _twist
    (send _twist :serialization-length)
    ;; geometry_msgs/Vector3 _bw
    (send _bw :serialization-length)
    ;; geometry_msgs/Vector3 _ba
    (send _ba :serialization-length)
    ;; geometry_msgs/Pose _extrinsics
    (send _extrinsics :serialization-length)
    ;; geometry_msgs/Vector3 _accel_meas
    (send _accel_meas :serialization-length)
    ;; int32 _N
    4
    ;; float32[] _error_cov
    (* 4    (length _error_cov)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; geometry_msgs/Pose _pose
       (send _pose :serialize s)
     ;; geometry_msgs/Twist _twist
       (send _twist :serialize s)
     ;; geometry_msgs/Vector3 _bw
       (send _bw :serialize s)
     ;; geometry_msgs/Vector3 _ba
       (send _ba :serialize s)
     ;; geometry_msgs/Pose _extrinsics
       (send _extrinsics :serialize s)
     ;; geometry_msgs/Vector3 _accel_meas
       (send _accel_meas :serialize s)
     ;; int32 _N
       (write-long _N s)
     ;; float32[] _error_cov
     (write-long (length _error_cov) s)
     (dotimes (i (length _error_cov))
       (sys::poke (elt _error_cov i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; geometry_msgs/Pose _pose
     (send _pose :deserialize buf ptr-) (incf ptr- (send _pose :serialization-length))
   ;; geometry_msgs/Twist _twist
     (send _twist :deserialize buf ptr-) (incf ptr- (send _twist :serialization-length))
   ;; geometry_msgs/Vector3 _bw
     (send _bw :deserialize buf ptr-) (incf ptr- (send _bw :serialization-length))
   ;; geometry_msgs/Vector3 _ba
     (send _ba :deserialize buf ptr-) (incf ptr- (send _ba :serialization-length))
   ;; geometry_msgs/Pose _extrinsics
     (send _extrinsics :deserialize buf ptr-) (incf ptr- (send _extrinsics :serialization-length))
   ;; geometry_msgs/Vector3 _accel_meas
     (send _accel_meas :deserialize buf ptr-) (incf ptr- (send _accel_meas :serialization-length))
   ;; int32 _N
     (setq _N (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; float32[] _error_cov
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _error_cov (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _error_cov i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(setf (get snapstack_msgs::VioFilterState :md5sum-) "081bdc0c4d73ef878229dd1b85815934")
(setf (get snapstack_msgs::VioFilterState :datatype-) "snapstack_msgs/VioFilterState")
(setf (get snapstack_msgs::VioFilterState :definition-)
      "# VioFilterState.msg
std_msgs/Header header

geometry_msgs/Pose pose
geometry_msgs/Twist twist  # includes gyro measurement for rates
geometry_msgs/Vector3 bw
geometry_msgs/Vector3 ba
geometry_msgs/Pose extrinsics  # camera-to-IMU transform

geometry_msgs/Vector3 accel_meas  # (IMU-frame) raw accel measurement

int32 N  # error state dimension
float32[] error_cov  # NxN error covariance.
                     # For consistency, leading 21 terms are ordered [pos, vel, Rwb, bw, ba, Rcb, tcb].

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
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

================================================================================
MSG: geometry_msgs/Twist
# This expresses velocity in free space broken into its linear and angular parts.
Vector3  linear
Vector3  angular

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



(provide :snapstack_msgs/VioFilterState "081bdc0c4d73ef878229dd1b85815934")


