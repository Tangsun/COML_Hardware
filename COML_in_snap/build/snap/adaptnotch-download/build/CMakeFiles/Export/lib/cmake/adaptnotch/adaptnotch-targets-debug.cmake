#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "adaptnotch" for configuration "Debug"
set_property(TARGET adaptnotch APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(adaptnotch PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libadaptnotch.so"
  IMPORTED_SONAME_DEBUG "libadaptnotch.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS adaptnotch )
list(APPEND _IMPORT_CHECK_FILES_FOR_adaptnotch "${_IMPORT_PREFIX}/lib/libadaptnotch.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
