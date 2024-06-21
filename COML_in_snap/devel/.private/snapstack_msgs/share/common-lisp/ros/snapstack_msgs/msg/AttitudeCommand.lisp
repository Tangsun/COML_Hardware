; Auto-generated. Do not edit!


(cl:in-package snapstack_msgs-msg)


;//! \htmlinclude AttitudeCommand.msg.html

(cl:defclass <AttitudeCommand> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (q
    :reader q
    :initarg :q
    :type geometry_msgs-msg:Quaternion
    :initform (cl:make-instance 'geometry_msgs-msg:Quaternion))
   (w
    :reader w
    :initarg :w
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (F_W
    :reader F_W
    :initarg :F_W
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (power
    :reader power
    :initarg :power
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass AttitudeCommand (<AttitudeCommand>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <AttitudeCommand>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'AttitudeCommand)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name snapstack_msgs-msg:<AttitudeCommand> is deprecated: use snapstack_msgs-msg:AttitudeCommand instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <AttitudeCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:header-val is deprecated.  Use snapstack_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'q-val :lambda-list '(m))
(cl:defmethod q-val ((m <AttitudeCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:q-val is deprecated.  Use snapstack_msgs-msg:q instead.")
  (q m))

(cl:ensure-generic-function 'w-val :lambda-list '(m))
(cl:defmethod w-val ((m <AttitudeCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:w-val is deprecated.  Use snapstack_msgs-msg:w instead.")
  (w m))

(cl:ensure-generic-function 'F_W-val :lambda-list '(m))
(cl:defmethod F_W-val ((m <AttitudeCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:F_W-val is deprecated.  Use snapstack_msgs-msg:F_W instead.")
  (F_W m))

(cl:ensure-generic-function 'power-val :lambda-list '(m))
(cl:defmethod power-val ((m <AttitudeCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:power-val is deprecated.  Use snapstack_msgs-msg:power instead.")
  (power m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <AttitudeCommand>) ostream)
  "Serializes a message object of type '<AttitudeCommand>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'q) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'w) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'F_W) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'power) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <AttitudeCommand>) istream)
  "Deserializes a message object of type '<AttitudeCommand>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'q) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'w) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'F_W) istream)
    (cl:setf (cl:slot-value msg 'power) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<AttitudeCommand>)))
  "Returns string type for a message object of type '<AttitudeCommand>"
  "snapstack_msgs/AttitudeCommand")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'AttitudeCommand)))
  "Returns string type for a message object of type 'AttitudeCommand"
  "snapstack_msgs/AttitudeCommand")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<AttitudeCommand>)))
  "Returns md5sum for a message object of type '<AttitudeCommand>"
  "06c299a6cb1b04a10cbd6ff8589ed1f9")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'AttitudeCommand)))
  "Returns md5sum for a message object of type 'AttitudeCommand"
  "06c299a6cb1b04a10cbd6ff8589ed1f9")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<AttitudeCommand>)))
  "Returns full string definition for message of type '<AttitudeCommand>"
  (cl:format cl:nil "Header header~%geometry_msgs/Quaternion q      # desired attitude~%geometry_msgs/Vector3 w         # desired angular rates~%geometry_msgs/Vector3 F_W       # desired force (expr in world frame)~%bool power # true if motors should be able to spin~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'AttitudeCommand)))
  "Returns full string definition for message of type 'AttitudeCommand"
  (cl:format cl:nil "Header header~%geometry_msgs/Quaternion q      # desired attitude~%geometry_msgs/Vector3 w         # desired angular rates~%geometry_msgs/Vector3 F_W       # desired force (expr in world frame)~%bool power # true if motors should be able to spin~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <AttitudeCommand>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'q))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'w))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'F_W))
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <AttitudeCommand>))
  "Converts a ROS message object to a list"
  (cl:list 'AttitudeCommand
    (cl:cons ':header (header msg))
    (cl:cons ':q (q msg))
    (cl:cons ':w (w msg))
    (cl:cons ':F_W (F_W msg))
    (cl:cons ':power (power msg))
))
