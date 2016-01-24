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
set(${PROJECT_NAME}_MINOR_VERSION 04)
set(${PROJECT_NAME}_PATCH_VERSION 04)
set(min_req_mtca4u-deviceaccess_version 00.04)

#
# Build up the project version name
set(${PROJECT_NAME}_VERSION
  ${${PROJECT_NAME}_MAJOR_VERSION}.${${PROJECT_NAME}_MINOR_VERSION}.${${PROJECT_NAME}_PATCH_VERSION})

#
# other variables 
set(boost_python_core_module "mtca4udeviceaccess")
set(python_module "mtca4u.py")
set(sphinx_build_confg_in "${PROJECT_SOURCE_DIR}/cmake/conf.py.in")
set(location_of_rst_source_files "${PROJECT_SOURCE_DIR}/doc")
set(location_of_built_html_files "${PROJECT_BINARY_DIR}/html")
set(sphinx_build_config_file "${PROJECT_BINARY_DIR}/conf.py")
set(author "MSK")

# Check if project dependencies are met
find_package(Boost REQUIRED COMPONENTS python)
find_package(PythonLibs REQUIRED)
find_package(mtca4u-deviceaccess ${min_req_mtca4u-deviceaccess_version} REQUIRED)
find_package(NumPy REQUIRED)
#

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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -Weffc++ -Wextra -Wshadow -pedantic -Wuninitialized")

#
# Set header file locations
include_directories(SYSTEM ${Boost_INCLUDE_DIRS} 
                            ${PYTHON_INCLUDE_DIR}
			    ${NUMPY_INCLUDE_DIRS}
                            ${mtca4u-deviceaccess_INCLUDE_DIRS})

#
# Define the build target for python module 
add_library(${boost_python_core_module} SHARED deviceaccessPython.cc 
                                         MultiplexedDataAccessorWrapper.cc
                                         CreateDevice.cc
                                         DeviceAccess.cc
                                         MuxDataAccessor.cc
                                         RegisterAccessor.cc)
                                         
target_link_libraries(${boost_python_core_module} ${mtca4u-deviceaccess_LIBRARIES}
                                            ${PYTHON_LIBRARIES} 
                                            ${Boost_LIBRARIES})
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
      CONFIGURE_FILE(cmake/Makefile.coverage.in
        ${PROJECT_BINARY_DIR}/Makefile.coverage @ONLY)
      ADD_CUSTOM_TARGET(coverage
        make -f Makefile.coverage
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating test coverage documentation" VERBATIM
      )
ENDIF(CMAKE_BUILD_TYPE STREQUAL "Debug")

#
# Make a list of the header files u would want to distribute with the dev
# package
set(headers_for_devpackage ${CMAKE_SOURCE_DIR}/HelperFunctions.h 
                           ${CMAKE_SOURCE_DIR}/PythonExceptions.h)
string(TOLOWER include/${PROJECT_NAME} header_install_location)

# These are cmake modules that would be installed to enable find_package command
# for this library on the target system
configure_file(cmake/Config.cmake.in
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake")
configure_file(cmake/ConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake")

# Installation targets for:
# - Pure python module
# - c++ python module
# - Headers fro the dev package
# - cmake modules for enabling find_package on this project 
install(FILES ${PROJECT_BINARY_DIR}/${python_module}
        DESTINATION lib/python2.7/dist-packages)
# do not remove runtime path to libmtca-deviceaccess location  from
# ${boost_python_core_module}  when installing
set_property(TARGET ${boost_python_core_module} 
	PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS ${boost_python_core_module}
        LIBRARY DESTINATION lib/python2.7/dist-packages)
install(FILES ${headers_for_devpackage}
        DESTINATION ${header_install_location})
install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" 
              "${CMAKE_SOURCE_DIR}/cmake/Modules/Find${PROJECT_NAME}.cmake"
        DESTINATION . 
        COMPONENT dev)

# cmake instructions for packaging; picked from
# ${CMAKE_SOURCE_DIR}/cmake/Modules, which has been appended to the 
# CMAKE_MODULE_PATH:
include(prepare_debian_package)
