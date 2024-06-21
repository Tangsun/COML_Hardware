
(cl:in-package :asdf)

(defsystem "behavior_selector-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "MissionModeChange" :depends-on ("_package_MissionModeChange"))
    (:file "_package_MissionModeChange" :depends-on ("_package"))
  ))