;; Auto-generated. Do not edit!


(when (boundp 'snapstack_msgs::TimeFilter)
  (if (not (find-package "SNAPSTACK_MSGS"))
    (make-package "SNAPSTACK_MSGS"))
  (shadow 'TimeFilter (find-package "SNAPSTACK_MSGS")))
(unless (find-package "SNAPSTACK_MSGS::TIMEFILTER")
  (make-package "SNAPSTACK_MSGS::TIMEFILTER"))

(in-package "ROS")
;;//! \htmlinclude TimeFilter.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass snapstack_msgs::TimeFilter
  :super ros::object
  :slots (_header _dt _delayed_dt _skipped _upper _lower ))

(defmethod snapstack_msgs::TimeFilter
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:dt __dt) 0.0)
    ((:delayed_dt __delayed_dt) 0.0)
    ((:skipped __skipped) nil)
    ((:upper __upper) 0.0)
    ((:lower __lower) 0.0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _dt (float __dt))
   (setq _delayed_dt (float __delayed_dt))
   (setq _skipped __skipped)
   (setq _upper (float __upper))
   (setq _lower (float __lower))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:dt
   (&optional __dt)
   (if __dt (setq _dt __dt)) _dt)
  (:delayed_dt
   (&optional __delayed_dt)
   (if __delayed_dt (setq _delayed_dt __delayed_dt)) _delayed_dt)
  (:skipped
   (&optional (__skipped :null))
   (if (not (eq __skipped :null)) (setq _skipped __skipped)) _skipped)
  (:upper
   (&optional __upper)
   (if __upper (setq _upper __upper)) _upper)
  (:lower
   (&optional __lower)
   (if __lower (setq _lower __lower)) _lower)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; float32 _dt
    4
    ;; float32 _delayed_dt
    4
    ;; bool _skipped
    1
    ;; float32 _upper
    4
    ;; float32 _lower
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; float32 _dt
       (sys::poke _dt (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _delayed_dt
       (sys::poke _delayed_dt (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; bool _skipped
       (if _skipped (write-byte -1 s) (write-byte 0 s))
     ;; float32 _upper
       (sys::poke _upper (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _lower
       (sys::poke _lower (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; float32 _dt
     (setq _dt (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _delayed_dt
     (setq _delayed_dt (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; bool _skipped
     (setq _skipped (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float32 _upper
     (setq _upper (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _lower
     (setq _lower (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get snapstack_msgs::TimeFilter :md5sum-) "ac265f085f2218e2759385a5695df9af")
(setf (get snapstack_msgs::TimeFilter :datatype-) "snapstack_msgs/TimeFilter")
(setf (get snapstack_msgs::TimeFilter :definition-)
      "# TimeFilter.msg

Header header
float32 dt  	# dt recorded in time stamp
float32 delayed_dt  # dt stamped when arrived onboard
bool skipped	# is msg skipped?
float32 upper 	# upper bound
float32 lower 	# lower bound
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

")



(provide :snapstack_msgs/TimeFilter "ac265f085f2218e2759385a5695df9af")


