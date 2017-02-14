cmake_minimum_required(VERSION 2.8)

project(mtca4uPy)
#==============================================================================#
# CMake modules needed for things to work:

# This location has some of the modules that we are using below
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
include(helperMacros)

#==============================================================================#


# version Information for the project
# bump up the version numbers in this section when creating new tags 
set(${PROJECT_NAME}_MAJOR_VERSION 00)
set(${PROJECT_NAME}_MINOR_VERSION 09)
set(${PROJECT_NAME}_PATCH_VERSION 01)
set(min_req_mtca4u-deviceaccess_version 00.19)

#
# Build up the project version name
set(${PROJECT_NAME}_VERSION
  ${${PROJECT_NAME}_MAJOR_VERSION}.${${PROJECT_NAME}_MINOR_VERSION}.${${PROJECT_NAME}_PATCH_VERSION})
set(${PROJECT_NAME}_SOVERSION
  ${${PROJECT_NAME}_MAJOR_VERSION}.${${PROJECT_NAME}_MINOR_VERSION})

# Set the build type to Release if none is specified
# Force it into Release if "None" is specified (needed to overrule dkpg_buildpackage)
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "None")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "None")

#
# other variables 
set(boost_python_core_module "mtca4udeviceaccess")
set(python_module "mtca4u.py")
set(sphinx_build_confg_in "${PROJECT_SOURCE_DIR}/cmake/conf.py.in")
set(location_of_rst_source_files "${PROJECT_SOURCE_DIR}/doc")
set(location_of_built_html_files "${PROJECT_BINARY_DIR}/doc/html")
set(sphinx_build_config_file "${PROJECT_BINARY_DIR}/conf.py")
set(author "MSK")

# Search for BOOST python library
find_package(Boost REQUIRED COMPONENTS python)

# Search for PythonLibs
find_package(PythonLibs 2.7 REQUIRED)

# Search for NumPy
find_package(NumPy REQUIRED)


# Search for DeviceAccess
find_package(mtca4u-deviceaccess ${min_req_mtca4u-deviceaccess_version} REQUIRED)
set(${PROJECT_NAME}_CXX_FLAGS "${${PROJECT_NAME}_CXX_FLAGS} ${mtca4u-deviceaccess_CXX_FLAGS}")

#
# add target for documentation if sphinx v1.3+ is available. 
CHECK_FOR_SPHINX()
if(SUPPORTED_SPHINX_VERSION_AVAILABLE)
  ADD_HTML_DOCUMENTATION_SUPPORT()
else()
  message(STATUS "Html doumentation support will not be enabled")
  message(STATUS "Please install working sphinx-doc version (1.3.0+) for html documentation support")
endif(SUPPORTED_SPHINX_VERSION_AVAILABLE)

#
# Set the compiler flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${${PROJECT_NAME}_CXX_FLAGS} -Wall -fPIC -Weffc++ -Wextra -Wshadow -pedantic -Wuninitialized")


#
# Reference: 
#   https://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html
#
# Defing NPY_NO_DEPRECATED_API as NPY_1_11_API_VERSION, should ensure that the
# compilation does not go through if the code uses API's marked as deprecated in
# numpy v1.11's documentation.

add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

#
# Set header file locations
include_directories(SYSTEM ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIRS} ${mtca4u-deviceaccess_INCLUDE_DIRS})

#
# Define the build target for python module 
add_library(${boost_python_core_module} SHARED deviceaccessPython.cc 
                                               CreateDevice.cc
                                               DeviceAccess.cc
                                               MuxDataAccessor.cc
                                               RegisterAccessor.cc)
                                         
target_link_libraries(${boost_python_core_module} ${mtca4u-deviceaccess_LIBRARIES} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
# donot prefix lib to the generated so:
set_target_properties(${boost_python_core_module} PROPERTIES PREFIX "")


# Copy the pure python module to the build directory. This is needed by the
# tests. Packaging looks for it in this location. The version information in the
# pure python module is filled out by cmake(configure_file). Ideally the python
# module name should end with a .py.in. However not sticking to this convention.
# The Module will end with .py though it will be processed by configure_file.
# Reason: would like to have py editor support on the file in the source
# directory.
configure_file( ${python_module}  ${PROJECT_BINARY_DIR})

#
# set up test cases
ENABLE_TESTING()
#
# Copy the test scripts to the build directory:
COPY_CONTENT_TO_BUILD_DIR("${CMAKE_SOURCE_DIR}/tests;${CMAKE_SOURCE_DIR}/tests/deviceInformation")
configure_file( ${CMAKE_SOURCE_DIR}/cmake/versionnumbers.py.in  ${PROJECT_BINARY_DIR}/versionnumbers.py)
#
# Add the scripts as ctests.
FILE(GLOB location_of_script_files 
    "${PROJECT_BINARY_DIR}/tests/*.py")
ADD_SCRIPTS_AS_TESTS( "${location_of_script_files}" )

#
# For now coverage is enabled in the debug Mode only
IF(CMAKE_BUILD_TYPE STREQUAL "Debug")

  #
  # Enable gdb and coverage report in the debug configuration.
  # (-DCMAKE_BUILD_TYPE=Debug)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 --coverage")

  #
  # Enable code coverage when cmake build configured -DCMAKE_BUILD_TYPE=Debug.
  # lcov gets run by the target defined in Makefile.coverage
  CONFIGURE_FILE(cmake/Makefile.coverage.in ${PROJECT_BINARY_DIR}/Makefile.coverage @ONLY)
  ADD_CUSTOM_TARGET(coverage make -f Makefile.coverage WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    COMMENT "Generating test coverage documentation" VERBATIM)

ENDIF(CMAKE_BUILD_TYPE STREQUAL "Debug")

#
# Make a list of the header files u would want to distribute with the dev
# package
set(headers_for_devpackage ${CMAKE_SOURCE_DIR}/HelperFunctions.h 
                           ${CMAKE_SOURCE_DIR}/PythonExceptions.h)
string(TOLOWER include/mtca4u/${PROJECT_NAME} header_install_location)

#Create the config files by filling the correct variables into the template (*.cmake.in).
#All variables @VARIABLE@ are replaced with the current values, but they have to be set first....
#Only the reference header is needed. The other variables are for dependencies of this library,
#but there are none.
set(${PROJECT_NAME}_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include/mtca4u
                                  ${Boost_INCLUDE_DIRS}
                                  ${PYTHON_INCLUDE_DIR}
                                  ${NUMPY_INCLUDE_DIRS}
                                  ${mtca4u-deviceaccess_INCLUDE_DIRS}")
set(${PROJECT_NAME}_LIBRARIES "${mtca4u-deviceaccess_LIBRARIES}
                               ${PYTHON_LIBRARIES}
                               ${NUMPY_LIBRARIES}
                               ${Boost_LIBRARIES}")
set(${PROJECT_NAME}_LIBRARY_DIRS "${CMAKE_INSTALL_PREFIX}/lib
                                  ${mtca4u-deviceaccess_LIBRARY_DIRS}
                                  ${PYTHON_LIBRARY_DIRS}
                                  ${NUMPY_LIBRARY_DIRS}
                                  ${Boost_LIBRARY_DIRS}")
# already set: ${PROJECT_NAME}_CXX_FLAGS
set(${PROJECT_NAME}_LINKER_FLAGS "-Wl,-rpath=${CMAKE_INSTALL_PREFIX}/lib,--enable-new-dtags")
include(${CMAKE_SOURCE_DIR}/cmake/create_cmake_config_files.cmake)

# Installation targets for:
# - Pure python module
# - c++ python module
# - Headers fro the dev package
# - cmake modules for enabling find_package on this project 
install(FILES ${PROJECT_BINARY_DIR}/${python_module} DESTINATION lib/python2.7/dist-packages)
# do not remove runtime path to libmtca-deviceaccess location  from
# ${boost_python_core_module}  when installing
set_property(TARGET ${boost_python_core_module} PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS ${boost_python_core_module} LIBRARY DESTINATION lib/python2.7/dist-packages)
install(FILES ${headers_for_devpackage} DESTINATION ${header_install_location})

# cmake instructions for packaging; picked from
# ${CMAKE_SOURCE_DIR}/cmake/Modules, which has been appended to the 
# CMAKE_MODULE_PATH:
include(prepare_debian_package)
