;; Auto-generated. Do not edit!


(when (boundp 'behavior_selector::MissionModeChange)
  (if (not (find-package "BEHAVIOR_SELECTOR"))
    (make-package "BEHAVIOR_SELECTOR"))
  (shadow 'MissionModeChange (find-package "BEHAVIOR_SELECTOR")))
(unless (find-package "BEHAVIOR_SELECTOR::MISSIONMODECHANGE")
  (make-package "BEHAVIOR_SELECTOR::MISSIONMODECHANGE"))
(unless (find-package "BEHAVIOR_SELECTOR::MISSIONMODECHANGEREQUEST")
  (make-package "BEHAVIOR_SELECTOR::MISSIONMODECHANGEREQUEST"))
(unless (find-package "BEHAVIOR_SELECTOR::MISSIONMODECHANGERESPONSE")
  (make-package "BEHAVIOR_SELECTOR::MISSIONMODECHANGERESPONSE"))

(in-package "ROS")





(intern "*START*" (find-package "BEHAVIOR_SELECTOR::MISSIONMODECHANGEREQUEST"))
(shadow '*START* (find-package "BEHAVIOR_SELECTOR::MISSIONMODECHANGEREQUEST"))
(defconstant behavior_selector::MissionModeChangeRequest::*START* 1)
(intern "*END*" (find-package "BEHAVIOR_SELECTOR::MISSIONMODECHANGEREQUEST"))
(shadow '*END* (find-package "BEHAVIOR_SELECTOR::MISSIONMODECHANGEREQUEST"))
(defconstant behavior_selector::MissionModeChangeRequest::*END* 2)
(intern "*KILL*" (find-package "BEHAVIOR_SELECTOR::MISSIONMODECHANGEREQUEST"))
(shadow '*KILL* (find-package "BEHAVIOR_SELECTOR::MISSIONMODECHANGEREQUEST"))
(defconstant behavior_selector::MissionModeChangeRequest::*KILL* 3)

(defun behavior_selector::MissionModeChangeRequest-to-symbol (const)
  (cond
        ((= const 1) 'behavior_selector::MissionModeChangeRequest::*START*)
        ((= const 2) 'behavior_selector::MissionModeChangeRequest::*END*)
        ((= const 3) 'behavior_selector::MissionModeChangeRequest::*KILL*)
        (t nil)))

(defclass behavior_selector::MissionModeChangeRequest
  :super ros::object
  :slots (_mode ))

(defmethod behavior_selector::MissionModeChangeRequest
  (:init
   (&key
    ((:mode __mode) 0)
    )
   (send-super :init)
   (setq _mode (round __mode))
   self)
  (:mode
   (&optional __mode)
   (if __mode (setq _mode __mode)) _mode)
  (:serialization-length
   ()
   (+
    ;; uint8 _mode
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _mode
       (write-byte _mode s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _mode
     (setq _mode (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;;
   self)
  )

(defclass behavior_selector::MissionModeChangeResponse
  :super ros::object
  :slots (_RECEIVED ))

(defmethod behavior_selector::MissionModeChangeResponse
  (:init
   (&key
    ((:RECEIVED __RECEIVED) nil)
    )
   (send-super :init)
   (setq _RECEIVED __RECEIVED)
   self)
  (:RECEIVED
   (&optional (__RECEIVED :null))
   (if (not (eq __RECEIVED :null)) (setq _RECEIVED __RECEIVED)) _RECEIVED)
  (:serialization-length
   ()
   (+
    ;; bool _RECEIVED
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _RECEIVED
       (if _RECEIVED (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _RECEIVED
     (setq _RECEIVED (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass behavior_selector::MissionModeChange
  :super ros::object
  :slots ())

(setf (get behavior_selector::MissionModeChange :md5sum-) "485ed44d9d9a6cafffa6f772d726a264")
(setf (get behavior_selector::MissionModeChange :datatype-) "behavior_selector/MissionModeChange")
(setf (get behavior_selector::MissionModeChange :request) behavior_selector::MissionModeChangeRequest)
(setf (get behavior_selector::MissionModeChange :response) behavior_selector::MissionModeChangeResponse)

(defmethod behavior_selector::MissionModeChangeRequest
  (:response () (instance behavior_selector::MissionModeChangeResponse :init)))

(setf (get behavior_selector::MissionModeChangeRequest :md5sum-) "485ed44d9d9a6cafffa6f772d726a264")
(setf (get behavior_selector::MissionModeChangeRequest :datatype-) "behavior_selector/MissionModeChangeRequest")
(setf (get behavior_selector::MissionModeChangeRequest :definition-)
      "uint8 mode
uint8 START  = 1
uint8 END    = 2
uint8 KILL   = 3
---
bool RECEIVED
")

(setf (get behavior_selector::MissionModeChangeResponse :md5sum-) "485ed44d9d9a6cafffa6f772d726a264")
(setf (get behavior_selector::MissionModeChangeResponse :datatype-) "behavior_selector/MissionModeChangeResponse")
(setf (get behavior_selector::MissionModeChangeResponse :definition-)
      "uint8 mode
uint8 START  = 1
uint8 END    = 2
uint8 KILL   = 3
---
bool RECEIVED
")



(provide :behavior_selector/MissionModeChange "485ed44d9d9a6cafffa6f772d726a264")


