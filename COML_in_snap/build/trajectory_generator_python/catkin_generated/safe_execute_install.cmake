execute_process(COMMAND "/home/sunbochen/Desktop/COML_Hardware/COML_in_snap/build/trajectory_generator_python/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/sunbochen/Desktop/COML_Hardware/COML_in_snap/build/trajectory_generator_python/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
