cmake_minimum_required(VERSION 3.16)

project(mtca4uPy)

# ==============================================================================#
# CMake modules needed for things to work:

# This location has some of the modules that we are using below
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake
  ${CMAKE_SOURCE_DIR}/cmake/Modules)

include(pythonBindingHelperMacros)

#
# We build for python3 by default
set(PYTHON3 TRUE CACHE BOOL "Builds modules for python3 if true and python2 if false")
initialize()

# ==============================================================================#

# version Information for the project
# bump up the version numbers in this section when creating new tags
set(${PROJECT_NAME}_MAJOR_VERSION 03)
set(${PROJECT_NAME}_MINOR_VERSION 02)
set(${PROJECT_NAME}_PATCH_VERSION 00)
set(min_req_chimeratk_version "03.09" CACHE STRING "")

#
# Build up the project version name
set(${PROJECT_NAME}_VERSION
  ${${PROJECT_NAME}_MAJOR_VERSION}.${${PROJECT_NAME}_MINOR_VERSION}.${${PROJECT_NAME}_PATCH_VERSION})
set(${PROJECT_NAME}_SOVERSION
  ${${PROJECT_NAME}_MAJOR_VERSION}.${${PROJECT_NAME}_MINOR_VERSION})

#
# variables used downstream
# ==============================================================================#
set(boost_python_core_module "_da_python_bindings")
set(legacy_python_module "mtca4u.py")
set(python_module "deviceaccess.py")
set(sphinx_build_confg_in "${PROJECT_SOURCE_DIR}/cmake/conf.py.in")
set(location_of_rst_source_files "${PROJECT_SOURCE_DIR}/doc")
set(location_of_built_html_files "${PROJECT_BINARY_DIR}/doc/html")
set(sphinx_build_config_file "${PROJECT_BINARY_DIR}/conf.py")
set(author "MSK")
get_python_interpreter_string(python_interpreter)
get_desired_python_major_release(py_major_release_num)

# Find all packages are needed for compilation.
# ==============================================================================#
find_package(PythonLibs ${py_major_release_num} REQUIRED)

find_package(Boost REQUIRED) # hack to get Boost_VERSION_STRING populated
get_boost_python_component_name(PYTHONLIBS_VERSION_STRING Boost_VERSION_STRING boost_python boost_numpy)
message("Boost Python aka ${boost_python} ")
message("Boost Numpy aka ${boost_numpy} ")
find_package(Boost REQUIRED COMPONENTS ${boost_python} ${boost_numpy})

# find_package(NumPy REQUIRED)
find_package(ChimeraTK-DeviceAccess ${min_req_chimeratk_version} REQUIRED)

# ==============================================================================#
INCLUDE(${CMAKE_SOURCE_DIR}/cmake/set_default_build_to_release.cmake)

INCLUDE(${CMAKE_SOURCE_DIR}/cmake/set_default_flags.cmake)

#
# Reference:
# https://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html
#
# Defing NPY_NO_DEPRECATED_API as NPY_1_11_API_VERSION, should ensure that the
# compilation does not go through if the code uses API's marked as deprecated in
# numpy v1.7's documentation.
add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

#
# Set header file locations
include_directories(SYSTEM ${Boost_INCLUDE_DIR}
  ${PYTHON_INCLUDE_DIR}
  ${NUMPY_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR})

# ==============================================================================#
# Only add the backend subdirectory after having all dependencies and flags
# add_subdirectory(tests/backend)

#
# add target for documentation if sphinx v1.3+ is available.
CHECK_FOR_SPHINX()

if(SUPPORTED_SPHINX_VERSION_AVAILABLE)
  ADD_HTML_DOCUMENTATION_SUPPORT()
else()
  message(STATUS "Html doumentation support will not be enabled")
  message(STATUS "Please install working sphinx-doc version (1.3.0+) for html documentation support (the package may be called python3-sphinx)")
endif(SUPPORTED_SPHINX_VERSION_AVAILABLE)

#
# Define the build target for python module
add_library(${boost_python_core_module} SHARED deviceaccessPython.cc
  PythonModuleMethods.cc
  CreateDevice.cc
  Register.cc
  DeviceAccess.cc)

if(ENABLE_TSAN)
  SET(TSAN_LIBS "tsan")
endif()

target_link_libraries(${boost_python_core_module} ${TSAN_LIBS}
  ChimeraTK::ChimeraTK-DeviceAccess
  ${PYTHON_LIBRARIES}
  ${Boost_LIBRARIES}
  Boost::${boost_python}
  Boost::${boost_numpy} ${TSAN_LIBS})

#
# Generate boost test executable
add_library(testmodule SHARED
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/testModule.cpp")
target_link_libraries(testmodule ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} Boost::${boost_python})

# Get numpy dir from python, if not set externally:
set(numpy_include_dirs "" CACHE PATH "Path to numpy headers")

if(NOT numpy_include_dirs)
execute_process(COMMAND "${python_interpreter}" -c "import numpy; print(numpy.get_include())"
                OUTPUT_VARIABLE numpy_include_dirs
                OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
message(STATUS "Numpy header files in: ${numpy_include_dirs}")

target_include_directories(testmodule PUBLIC "${numpy_include_dirs}")

# donot prefix lib to the generated so:
set_target_properties(testmodule PROPERTIES PREFIX "")
set_target_properties(${boost_python_core_module} PROPERTIES PREFIX "")

# Copy the pure python module to the build directory. This is needed by the
# tests. Packaging looks for it in this location. The version information in the
# pure python module is filled out by cmake(configure_file). Ideally the python
# module name should end with a .py.in. However not sticking to this convention.
# The Module will end with .py though it will be processed by configure_file.
# Reason: would like to have py editor support on the file in the source
# directory.
configure_file(${legacy_python_module} ${PROJECT_BINARY_DIR})
configure_file(${python_module} ${PROJECT_BINARY_DIR})

#
# set up test cases
ENABLE_TESTING()

#
# Copy the test scripts to the build directory:
COPY_CONTENT_TO_BUILD_DIR("${CMAKE_SOURCE_DIR}/tests;${CMAKE_SOURCE_DIR}/tests/deviceInformation")
configure_file(${CMAKE_SOURCE_DIR}/cmake/versionnumbers.py.in ${PROJECT_BINARY_DIR}/versionnumbers.py)

#
# Add the scripts as ctests.
FILE(GLOB location_of_script_files
  "${PROJECT_BINARY_DIR}/tests/*.py")
ADD_SCRIPTS_AS_TESTS("${location_of_script_files}")

# Push type tests might run indefinitely if no timeout is set and script is broken
set_tests_properties(testPushType PROPERTIES TIMEOUT 5)

#
# enable coverage in debug Mode(CMAKE_BUILD_TYPE == "Debug").
include(enable_code_coverage_report)

# do not remove runtime path to libmtca-deviceaccess location  from
# ${boost_python_core_module}  when installing
set_property(TARGET ${boost_python_core_module}
  PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

get_python_module_install_path(PYTHONLIBS_VERSION_STRING install_path)
message(STATUS "Install Path: ${install_path}")

# Installation targets for:
# - Pure python module
# - c++ python module
# - Headers for the dev package
set(python_module_path ${install_path})
install(FILES ${PROJECT_BINARY_DIR}/${python_module}
  DESTINATION ${python_module_path})
install(FILES ${PROJECT_BINARY_DIR}/${legacy_python_module}
  DESTINATION ${install_path})
install(TARGETS ${boost_python_core_module}
  LIBRARY DESTINATION ${install_path})
