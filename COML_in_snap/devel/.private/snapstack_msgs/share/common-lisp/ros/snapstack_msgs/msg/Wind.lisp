; Auto-generated. Do not edit!


(cl:in-package snapstack_msgs-msg)


;//! \htmlinclude Wind.msg.html

(cl:defclass <Wind> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (w_nominal
    :reader w_nominal
    :initarg :w_nominal
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (w_gust
    :reader w_gust
    :initarg :w_gust
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3)))
)

(cl:defclass Wind (<Wind>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Wind>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Wind)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name snapstack_msgs-msg:<Wind> is deprecated: use snapstack_msgs-msg:Wind instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <Wind>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:header-val is deprecated.  Use snapstack_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'w_nominal-val :lambda-list '(m))
(cl:defmethod w_nominal-val ((m <Wind>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:w_nominal-val is deprecated.  Use snapstack_msgs-msg:w_nominal instead.")
  (w_nominal m))

(cl:ensure-generic-function 'w_gust-val :lambda-list '(m))
(cl:defmethod w_gust-val ((m <Wind>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:w_gust-val is deprecated.  Use snapstack_msgs-msg:w_gust instead.")
  (w_gust m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Wind>) ostream)
  "Serializes a message object of type '<Wind>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'w_nominal) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'w_gust) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Wind>) istream)
  "Deserializes a message object of type '<Wind>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'w_nominal) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'w_gust) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Wind>)))
  "Returns string type for a message object of type '<Wind>"
  "snapstack_msgs/Wind")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Wind)))
  "Returns string type for a message object of type 'Wind"
  "snapstack_msgs/Wind")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Wind>)))
  "Returns md5sum for a message object of type '<Wind>"
  "43d6adbcc621b86e3d87893f8b41e553")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Wind)))
  "Returns md5sum for a message object of type 'Wind"
  "43d6adbcc621b86e3d87893f8b41e553")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Wind>)))
  "Returns full string definition for message of type '<Wind>"
  (cl:format cl:nil "Header header~%geometry_msgs/Vector3 w_nominal~%geometry_msgs/Vector3 w_gust~%~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Wind)))
  "Returns full string definition for message of type 'Wind"
  (cl:format cl:nil "Header header~%geometry_msgs/Vector3 w_nominal~%geometry_msgs/Vector3 w_gust~%~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Wind>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'w_nominal))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'w_gust))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Wind>))
  "Converts a ROS message object to a list"
  (cl:list 'Wind
    (cl:cons ':header (header msg))
    (cl:cons ':w_nominal (w_nominal msg))
    (cl:cons ':w_gust (w_gust msg))
))
