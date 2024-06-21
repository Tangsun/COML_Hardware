cmake_minimum_required(VERSION 3.1)
project(adaptnotch-download NONE)

include(ExternalProject)
ExternalProject_Add(adaptnotch
  GIT_REPOSITORY    https://github.com/plusk01/adaptive-gyro-filtering
  GIT_TAG           master
  SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/src"
  BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/build"
  CONFIGURE_COMMAND
      cmake ${CMAKE_CURRENT_BINARY_DIR}/src
          -DCMAKE_BUILD_TYPE=Release
          -DBUILD_EXAPPS=OFF
          -DCMAKE_INSTALL_PREFIX=/home/sunbochen/Desktop/COML_Hardware/COML_in_snap/devel/.private/snap
  BUILD_COMMAND
      make -j
  INSTALL_COMMAND
      make install
  TEST_COMMAND      ""
)
