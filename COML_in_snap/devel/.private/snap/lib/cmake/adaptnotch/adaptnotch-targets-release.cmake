#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "adaptnotch" for configuration "Release"
set_property(TARGET adaptnotch APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(adaptnotch PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libadaptnotch.so"
  IMPORTED_SONAME_RELEASE "libadaptnotch.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS adaptnotch )
list(APPEND _IMPORT_CHECK_FILES_FOR_adaptnotch "${_IMPORT_PREFIX}/lib/libadaptnotch.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
