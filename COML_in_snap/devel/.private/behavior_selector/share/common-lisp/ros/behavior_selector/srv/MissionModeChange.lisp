; Auto-generated. Do not edit!


(cl:in-package behavior_selector-srv)


;//! \htmlinclude MissionModeChange-request.msg.html

(cl:defclass <MissionModeChange-request> (roslisp-msg-protocol:ros-message)
  ((mode
    :reader mode
    :initarg :mode
    :type cl:fixnum
    :initform 0))
)

(cl:defclass MissionModeChange-request (<MissionModeChange-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <MissionModeChange-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'MissionModeChange-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name behavior_selector-srv:<MissionModeChange-request> is deprecated: use behavior_selector-srv:MissionModeChange-request instead.")))

(cl:ensure-generic-function 'mode-val :lambda-list '(m))
(cl:defmethod mode-val ((m <MissionModeChange-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader behavior_selector-srv:mode-val is deprecated.  Use behavior_selector-srv:mode instead.")
  (mode m))
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql '<MissionModeChange-request>)))
    "Constants for message type '<MissionModeChange-request>"
  '((:START . 1)
    (:END . 2)
    (:KILL . 3))
)
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql 'MissionModeChange-request)))
    "Constants for message type 'MissionModeChange-request"
  '((:START . 1)
    (:END . 2)
    (:KILL . 3))
)
(cl:defmethod roslisp-msg-protocol:serialize ((msg <MissionModeChange-request>) ostream)
  "Serializes a message object of type '<MissionModeChange-request>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'mode)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <MissionModeChange-request>) istream)
  "Deserializes a message object of type '<MissionModeChange-request>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'mode)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<MissionModeChange-request>)))
  "Returns string type for a service object of type '<MissionModeChange-request>"
  "behavior_selector/MissionModeChangeRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MissionModeChange-request)))
  "Returns string type for a service object of type 'MissionModeChange-request"
  "behavior_selector/MissionModeChangeRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<MissionModeChange-request>)))
  "Returns md5sum for a message object of type '<MissionModeChange-request>"
  "485ed44d9d9a6cafffa6f772d726a264")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'MissionModeChange-request)))
  "Returns md5sum for a message object of type 'MissionModeChange-request"
  "485ed44d9d9a6cafffa6f772d726a264")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<MissionModeChange-request>)))
  "Returns full string definition for message of type '<MissionModeChange-request>"
  (cl:format cl:nil "uint8 mode~%uint8 START  = 1~%uint8 END    = 2~%uint8 KILL   = 3~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'MissionModeChange-request)))
  "Returns full string definition for message of type 'MissionModeChange-request"
  (cl:format cl:nil "uint8 mode~%uint8 START  = 1~%uint8 END    = 2~%uint8 KILL   = 3~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <MissionModeChange-request>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <MissionModeChange-request>))
  "Converts a ROS message object to a list"
  (cl:list 'MissionModeChange-request
    (cl:cons ':mode (mode msg))
))
;//! \htmlinclude MissionModeChange-response.msg.html

(cl:defclass <MissionModeChange-response> (roslisp-msg-protocol:ros-message)
  ((RECEIVED
    :reader RECEIVED
    :initarg :RECEIVED
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass MissionModeChange-response (<MissionModeChange-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <MissionModeChange-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'MissionModeChange-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name behavior_selector-srv:<MissionModeChange-response> is deprecated: use behavior_selector-srv:MissionModeChange-response instead.")))

(cl:ensure-generic-function 'RECEIVED-val :lambda-list '(m))
(cl:defmethod RECEIVED-val ((m <MissionModeChange-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader behavior_selector-srv:RECEIVED-val is deprecated.  Use behavior_selector-srv:RECEIVED instead.")
  (RECEIVED m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <MissionModeChange-response>) ostream)
  "Serializes a message object of type '<MissionModeChange-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'RECEIVED) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <MissionModeChange-response>) istream)
  "Deserializes a message object of type '<MissionModeChange-response>"
    (cl:setf (cl:slot-value msg 'RECEIVED) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<MissionModeChange-response>)))
  "Returns string type for a service object of type '<MissionModeChange-response>"
  "behavior_selector/MissionModeChangeResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MissionModeChange-response)))
  "Returns string type for a service object of type 'MissionModeChange-response"
  "behavior_selector/MissionModeChangeResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<MissionModeChange-response>)))
  "Returns md5sum for a message object of type '<MissionModeChange-response>"
  "485ed44d9d9a6cafffa6f772d726a264")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'MissionModeChange-response)))
  "Returns md5sum for a message object of type 'MissionModeChange-response"
  "485ed44d9d9a6cafffa6f772d726a264")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<MissionModeChange-response>)))
  "Returns full string definition for message of type '<MissionModeChange-response>"
  (cl:format cl:nil "bool RECEIVED~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'MissionModeChange-response)))
  "Returns full string definition for message of type 'MissionModeChange-response"
  (cl:format cl:nil "bool RECEIVED~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <MissionModeChange-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <MissionModeChange-response>))
  "Converts a ROS message object to a list"
  (cl:list 'MissionModeChange-response
    (cl:cons ':RECEIVED (RECEIVED msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'MissionModeChange)))
  'MissionModeChange-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'MissionModeChange)))
  'MissionModeChange-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MissionModeChange)))
  "Returns string type for a service object of type '<MissionModeChange>"
  "behavior_selector/MissionModeChange")