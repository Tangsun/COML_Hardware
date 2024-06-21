; Auto-generated. Do not edit!


(cl:in-package snapstack_msgs-msg)


;//! \htmlinclude VioFilterState.msg.html

(cl:defclass <VioFilterState> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (pose
    :reader pose
    :initarg :pose
    :type geometry_msgs-msg:Pose
    :initform (cl:make-instance 'geometry_msgs-msg:Pose))
   (twist
    :reader twist
    :initarg :twist
    :type geometry_msgs-msg:Twist
    :initform (cl:make-instance 'geometry_msgs-msg:Twist))
   (bw
    :reader bw
    :initarg :bw
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (ba
    :reader ba
    :initarg :ba
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (extrinsics
    :reader extrinsics
    :initarg :extrinsics
    :type geometry_msgs-msg:Pose
    :initform (cl:make-instance 'geometry_msgs-msg:Pose))
   (accel_meas
    :reader accel_meas
    :initarg :accel_meas
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (N
    :reader N
    :initarg :N
    :type cl:integer
    :initform 0)
   (error_cov
    :reader error_cov
    :initarg :error_cov
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass VioFilterState (<VioFilterState>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <VioFilterState>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'VioFilterState)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name snapstack_msgs-msg:<VioFilterState> is deprecated: use snapstack_msgs-msg:VioFilterState instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <VioFilterState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:header-val is deprecated.  Use snapstack_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'pose-val :lambda-list '(m))
(cl:defmethod pose-val ((m <VioFilterState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:pose-val is deprecated.  Use snapstack_msgs-msg:pose instead.")
  (pose m))

(cl:ensure-generic-function 'twist-val :lambda-list '(m))
(cl:defmethod twist-val ((m <VioFilterState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:twist-val is deprecated.  Use snapstack_msgs-msg:twist instead.")
  (twist m))

(cl:ensure-generic-function 'bw-val :lambda-list '(m))
(cl:defmethod bw-val ((m <VioFilterState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:bw-val is deprecated.  Use snapstack_msgs-msg:bw instead.")
  (bw m))

(cl:ensure-generic-function 'ba-val :lambda-list '(m))
(cl:defmethod ba-val ((m <VioFilterState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:ba-val is deprecated.  Use snapstack_msgs-msg:ba instead.")
  (ba m))

(cl:ensure-generic-function 'extrinsics-val :lambda-list '(m))
(cl:defmethod extrinsics-val ((m <VioFilterState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:extrinsics-val is deprecated.  Use snapstack_msgs-msg:extrinsics instead.")
  (extrinsics m))

(cl:ensure-generic-function 'accel_meas-val :lambda-list '(m))
(cl:defmethod accel_meas-val ((m <VioFilterState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:accel_meas-val is deprecated.  Use snapstack_msgs-msg:accel_meas instead.")
  (accel_meas m))

(cl:ensure-generic-function 'N-val :lambda-list '(m))
(cl:defmethod N-val ((m <VioFilterState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:N-val is deprecated.  Use snapstack_msgs-msg:N instead.")
  (N m))

(cl:ensure-generic-function 'error_cov-val :lambda-list '(m))
(cl:defmethod error_cov-val ((m <VioFilterState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:error_cov-val is deprecated.  Use snapstack_msgs-msg:error_cov instead.")
  (error_cov m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <VioFilterState>) ostream)
  "Serializes a message object of type '<VioFilterState>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'pose) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'twist) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'bw) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'ba) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'extrinsics) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'accel_meas) ostream)
  (cl:let* ((signed (cl:slot-value msg 'N)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'error_cov))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'error_cov))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <VioFilterState>) istream)
  "Deserializes a message object of type '<VioFilterState>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'pose) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'twist) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'bw) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'ba) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'extrinsics) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'accel_meas) istream)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'N) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'error_cov) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'error_cov)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<VioFilterState>)))
  "Returns string type for a message object of type '<VioFilterState>"
  "snapstack_msgs/VioFilterState")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'VioFilterState)))
  "Returns string type for a message object of type 'VioFilterState"
  "snapstack_msgs/VioFilterState")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<VioFilterState>)))
  "Returns md5sum for a message object of type '<VioFilterState>"
  "081bdc0c4d73ef878229dd1b85815934")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'VioFilterState)))
  "Returns md5sum for a message object of type 'VioFilterState"
  "081bdc0c4d73ef878229dd1b85815934")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<VioFilterState>)))
  "Returns full string definition for message of type '<VioFilterState>"
  (cl:format cl:nil "# VioFilterState.msg~%std_msgs/Header header~%~%geometry_msgs/Pose pose~%geometry_msgs/Twist twist  # includes gyro measurement for rates~%geometry_msgs/Vector3 bw~%geometry_msgs/Vector3 ba~%geometry_msgs/Pose extrinsics  # camera-to-IMU transform~%~%geometry_msgs/Vector3 accel_meas  # (IMU-frame) raw accel measurement~%~%int32 N  # error state dimension~%float32[] error_cov  # NxN error covariance.~%                     # For consistency, leading 21 terms are ordered [pos, vel, Rwb, bw, ba, Rcb, tcb].~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: geometry_msgs/Twist~%# This expresses velocity in free space broken into its linear and angular parts.~%Vector3  linear~%Vector3  angular~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'VioFilterState)))
  "Returns full string definition for message of type 'VioFilterState"
  (cl:format cl:nil "# VioFilterState.msg~%std_msgs/Header header~%~%geometry_msgs/Pose pose~%geometry_msgs/Twist twist  # includes gyro measurement for rates~%geometry_msgs/Vector3 bw~%geometry_msgs/Vector3 ba~%geometry_msgs/Pose extrinsics  # camera-to-IMU transform~%~%geometry_msgs/Vector3 accel_meas  # (IMU-frame) raw accel measurement~%~%int32 N  # error state dimension~%float32[] error_cov  # NxN error covariance.~%                     # For consistency, leading 21 terms are ordered [pos, vel, Rwb, bw, ba, Rcb, tcb].~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: geometry_msgs/Twist~%# This expresses velocity in free space broken into its linear and angular parts.~%Vector3  linear~%Vector3  angular~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <VioFilterState>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'pose))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'twist))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'bw))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'ba))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'extrinsics))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'accel_meas))
     4
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'error_cov) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <VioFilterState>))
  "Converts a ROS message object to a list"
  (cl:list 'VioFilterState
    (cl:cons ':header (header msg))
    (cl:cons ':pose (pose msg))
    (cl:cons ':twist (twist msg))
    (cl:cons ':bw (bw msg))
    (cl:cons ':ba (ba msg))
    (cl:cons ':extrinsics (extrinsics msg))
    (cl:cons ':accel_meas (accel_meas msg))
    (cl:cons ':N (N msg))
    (cl:cons ':error_cov (error_cov msg))
))
