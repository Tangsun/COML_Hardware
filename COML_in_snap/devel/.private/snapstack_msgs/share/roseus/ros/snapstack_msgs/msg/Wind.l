;; Auto-generated. Do not edit!


(when (boundp 'snapstack_msgs::Wind)
  (if (not (find-package "SNAPSTACK_MSGS"))
    (make-package "SNAPSTACK_MSGS"))
  (shadow 'Wind (find-package "SNAPSTACK_MSGS")))
(unless (find-package "SNAPSTACK_MSGS::WIND")
  (make-package "SNAPSTACK_MSGS::WIND"))

(in-package "ROS")
;;//! \htmlinclude Wind.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass snapstack_msgs::Wind
  :super ros::object
  :slots (_header _w_nominal _w_gust ))

(defmethod snapstack_msgs::Wind
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:w_nominal __w_nominal) (instance geometry_msgs::Vector3 :init))
    ((:w_gust __w_gust) (instance geometry_msgs::Vector3 :init))
    )
   (send-super :init)
   (setq _header __header)
   (setq _w_nominal __w_nominal)
   (setq _w_gust __w_gust)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:w_nominal
   (&rest __w_nominal)
   (if (keywordp (car __w_nominal))
       (send* _w_nominal __w_nominal)
     (progn
       (if __w_nominal (setq _w_nominal (car __w_nominal)))
       _w_nominal)))
  (:w_gust
   (&rest __w_gust)
   (if (keywordp (car __w_gust))
       (send* _w_gust __w_gust)
     (progn
       (if __w_gust (setq _w_gust (car __w_gust)))
       _w_gust)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; geometry_msgs/Vector3 _w_nominal
    (send _w_nominal :serialization-length)
    ;; geometry_msgs/Vector3 _w_gust
    (send _w_gust :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; geometry_msgs/Vector3 _w_nominal
       (send _w_nominal :serialize s)
     ;; geometry_msgs/Vector3 _w_gust
       (send _w_gust :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; geometry_msgs/Vector3 _w_nominal
     (send _w_nominal :deserialize buf ptr-) (incf ptr- (send _w_nominal :serialization-length))
   ;; geometry_msgs/Vector3 _w_gust
     (send _w_gust :deserialize buf ptr-) (incf ptr- (send _w_gust :serialization-length))
   ;;
   self)
  )

(setf (get snapstack_msgs::Wind :md5sum-) "43d6adbcc621b86e3d87893f8b41e553")
(setf (get snapstack_msgs::Wind :datatype-) "snapstack_msgs/Wind")
(setf (get snapstack_msgs::Wind :definition-)
      "Header header
geometry_msgs/Vector3 w_nominal
geometry_msgs/Vector3 w_gust


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



(provide :snapstack_msgs/Wind "43d6adbcc621b86e3d87893f8b41e553")


