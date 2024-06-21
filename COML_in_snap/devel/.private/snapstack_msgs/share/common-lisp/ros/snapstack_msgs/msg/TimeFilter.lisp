; Auto-generated. Do not edit!


(cl:in-package snapstack_msgs-msg)


;//! \htmlinclude TimeFilter.msg.html

(cl:defclass <TimeFilter> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (dt
    :reader dt
    :initarg :dt
    :type cl:float
    :initform 0.0)
   (delayed_dt
    :reader delayed_dt
    :initarg :delayed_dt
    :type cl:float
    :initform 0.0)
   (skipped
    :reader skipped
    :initarg :skipped
    :type cl:boolean
    :initform cl:nil)
   (upper
    :reader upper
    :initarg :upper
    :type cl:float
    :initform 0.0)
   (lower
    :reader lower
    :initarg :lower
    :type cl:float
    :initform 0.0))
)

(cl:defclass TimeFilter (<TimeFilter>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TimeFilter>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TimeFilter)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name snapstack_msgs-msg:<TimeFilter> is deprecated: use snapstack_msgs-msg:TimeFilter instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <TimeFilter>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:header-val is deprecated.  Use snapstack_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'dt-val :lambda-list '(m))
(cl:defmethod dt-val ((m <TimeFilter>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:dt-val is deprecated.  Use snapstack_msgs-msg:dt instead.")
  (dt m))

(cl:ensure-generic-function 'delayed_dt-val :lambda-list '(m))
(cl:defmethod delayed_dt-val ((m <TimeFilter>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:delayed_dt-val is deprecated.  Use snapstack_msgs-msg:delayed_dt instead.")
  (delayed_dt m))

(cl:ensure-generic-function 'skipped-val :lambda-list '(m))
(cl:defmethod skipped-val ((m <TimeFilter>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:skipped-val is deprecated.  Use snapstack_msgs-msg:skipped instead.")
  (skipped m))

(cl:ensure-generic-function 'upper-val :lambda-list '(m))
(cl:defmethod upper-val ((m <TimeFilter>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:upper-val is deprecated.  Use snapstack_msgs-msg:upper instead.")
  (upper m))

(cl:ensure-generic-function 'lower-val :lambda-list '(m))
(cl:defmethod lower-val ((m <TimeFilter>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader snapstack_msgs-msg:lower-val is deprecated.  Use snapstack_msgs-msg:lower instead.")
  (lower m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TimeFilter>) ostream)
  "Serializes a message object of type '<TimeFilter>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'dt))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'delayed_dt))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'skipped) 1 0)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'upper))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'lower))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TimeFilter>) istream)
  "Deserializes a message object of type '<TimeFilter>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'dt) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'delayed_dt) (roslisp-utils:decode-single-float-bits bits)))
    (cl:setf (cl:slot-value msg 'skipped) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'upper) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'lower) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TimeFilter>)))
  "Returns string type for a message object of type '<TimeFilter>"
  "snapstack_msgs/TimeFilter")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TimeFilter)))
  "Returns string type for a message object of type 'TimeFilter"
  "snapstack_msgs/TimeFilter")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TimeFilter>)))
  "Returns md5sum for a message object of type '<TimeFilter>"
  "ac265f085f2218e2759385a5695df9af")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TimeFilter)))
  "Returns md5sum for a message object of type 'TimeFilter"
  "ac265f085f2218e2759385a5695df9af")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TimeFilter>)))
  "Returns full string definition for message of type '<TimeFilter>"
  (cl:format cl:nil "# TimeFilter.msg~%~%Header header~%float32 dt  	# dt recorded in time stamp~%float32 delayed_dt  # dt stamped when arrived onboard~%bool skipped	# is msg skipped?~%float32 upper 	# upper bound~%float32 lower 	# lower bound~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TimeFilter)))
  "Returns full string definition for message of type 'TimeFilter"
  (cl:format cl:nil "# TimeFilter.msg~%~%Header header~%float32 dt  	# dt recorded in time stamp~%float32 delayed_dt  # dt stamped when arrived onboard~%bool skipped	# is msg skipped?~%float32 upper 	# upper bound~%float32 lower 	# lower bound~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TimeFilter>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4
     4
     1
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TimeFilter>))
  "Converts a ROS message object to a list"
  (cl:list 'TimeFilter
    (cl:cons ':header (header msg))
    (cl:cons ':dt (dt msg))
    (cl:cons ':delayed_dt (delayed_dt msg))
    (cl:cons ':skipped (skipped msg))
    (cl:cons ':upper (upper msg))
    (cl:cons ':lower (lower msg))
))
