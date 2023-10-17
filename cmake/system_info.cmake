### NATIVE FLAGS ###
if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  message(STATUS "Build is on x86_64 system.")
  list(APPEND MODYN_NATIVE_FLAGS "-march=native")
  list(APPEND MODYN_COMPILE_DEFINITIONS "MODYN_IS_X86=1")
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64" OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
  message(STATUS "Build is on ARM system.")
  list(APPEND MODYN_NATIVE_FLAGS "-mcpu=native")
  list(APPEND MODYN_COMPILE_DEFINITIONS "MODYN_IS_ARM=1")
else ()
  message(STATUS "Unsupported platform ${CMAKE_SYSTEM_PROCESSOR}. Not using any native flags.")
endif ()

list(APPEND MODYN_COMPILE_OPTIONS ${MODYN_NATIVE_FLAGS})
