cmake_minimum_required(VERSION 2.8)

project(mtca4uPy)
#==============================================================================#
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
#==============================================================================#

# version Information for the project
# bump up the version numbers in this section when creating new tags 
set(${PROJECT_NAME}_MAJOR_VERSION 02)
set(${PROJECT_NAME}_MINOR_VERSION 01)
set(${PROJECT_NAME}_PATCH_VERSION 00)
set(min_req_chimeratk_version "02.04" CACHE STRING "")

#
# Build up the project version name
set(${PROJECT_NAME}_VERSION
  ${${PROJECT_NAME}_MAJOR_VERSION}.${${PROJECT_NAME}_MINOR_VERSION}.${${PROJECT_NAME}_PATCH_VERSION})
set(${PROJECT_NAME}_SOVERSION
  ${${PROJECT_NAME}_MAJOR_VERSION}.${${PROJECT_NAME}_MINOR_VERSION})

#
# variables used downstream 
#==============================================================================#
set(boost_python_core_module "mtca4udeviceaccess")
set(python_module "mtca4u.py")
set(sphinx_build_confg_in "${PROJECT_SOURCE_DIR}/cmake/conf.py.in")
set(location_of_rst_source_files "${PROJECT_SOURCE_DIR}/doc")
set(location_of_built_html_files "${PROJECT_BINARY_DIR}/doc/html")
set(sphinx_build_config_file "${PROJECT_BINARY_DIR}/conf.py")
set(author "MSK")
get_python_interpreter_string(python_interpreter)
get_desired_python_major_release(py_major_release_num)

# Find all packages are needed for compilation.
#==============================================================================#
find_package(PythonLibs ${py_major_release_num} REQUIRED)

find_package(Boost REQUIRED) # hack to get Boost_VERSION_STRING populated
get_boost_python_component_name(PYTHONLIBS_VERSION_STRING Boost_VERSION_STRING boost_python)
find_package(Boost REQUIRED COMPONENTS ${boost_python})

find_package(NumPy REQUIRED)
find_package(ChimeraTK-DeviceAccess ${min_req_chimeratk_version} REQUIRED)
#==============================================================================#

INCLUDE(${CMAKE_SOURCE_DIR}/cmake/set_default_build_to_release.cmake)

#
# Set the compiler flags.
set(${PROJECT_NAME}_CXX_FLAGS 
   "${${PROJECT_NAME}_CXX_FLAGS} ${ChimeraTK-DeviceAccess_CXX_FLAGS}")

INCLUDE(${CMAKE_SOURCE_DIR}/cmake/set_default_flags.cmake)


#
# Reference: 
#   https://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html
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
                           ${CMAKE_SOURCE_DIR}
                           ${ChimeraTK-DeviceAccess_INCLUDE_DIRS})

#==============================================================================#
# Only add the backend subdirectory after having all dependencies and flags
add_subdirectory(tests/backend)

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
                                               CreateDevice.cc
                                               DeviceAccess.cc)

if(ENABLE_TSAN)
  SET(TSAN_LIBS "tsan")
endif()                                               
                                             
target_link_libraries(${boost_python_core_module} ${TSAN_LIBS}
                      ${ChimeraTK-DeviceAccess_LIBRARIES} 
                      ${PYTHON_LIBRARIES} 
                      ${Boost_LIBRARIES}
                      Boost::${boost_python} ${TSAN_LIBS})
                  
#
# Generate boost test executable
add_library( testmodule SHARED
               "${CMAKE_CURRENT_SOURCE_DIR}/tests/testModule.cpp" )
target_link_libraries(testmodule ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} Boost::${boost_python})


# donot prefix lib to the generated so:
set_target_properties(testmodule PROPERTIES PREFIX "")
set_target_properties(${boost_python_core_module} PROPERTIES PREFIX "")

# obey link flags from DeviceAccess (e.g. rpath)
set_target_properties(testmodule PROPERTIES LINK_FLAGS "${ChimeraTK-DeviceAccess_LINKER_FLAGS}")
set_target_properties(${boost_python_core_module}  PROPERTIES LINK_FLAGS "${ChimeraTK-DeviceAccess_LINKER_FLAGS}")


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
# enable coverage in debug Mode(CMAKE_BUILD_TYPE == "Debug").
include(enable_code_coverage_report)

#Create the config files by filling the correct variables into the template (*.cmake.in).
#All variables @VARIABLE@ are replaced with the current values, but they have to be set first....
#Only the reference header is needed. The other variables are for dependencies of this library,
#but there are none.
set(${PROJECT_NAME}_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include/mtca4u
                                  ${Boost_INCLUDE_DIR}
                                  ${PYTHON_INCLUDE_DIR}
                                  ${NUMPY_INCLUDE_DIRS}
                                  ${ChimeraTK-DeviceAccess_INCLUDE_DIRS}")
set(${PROJECT_NAME}_LIBRARIES "${ChimeraTK-DeviceAccess_LIBRARIES}
                               ${PYTHON_LIBRARIES}
                               ${NUMPY_LIBRARIES}
                               ${Boost_LIBRARIES}")
set(${PROJECT_NAME}_LIBRARY_DIRS "${CMAKE_INSTALL_PREFIX}/lib
                                  ${ChimeraTK-DeviceAccess_LIBRARY_DIRS}
                                  ${PYTHON_LIBRARY_DIRS}
                                  ${NUMPY_LIBRARY_DIRS}
                                  ${Boost_LIBRARY_DIRS}")
                              
# already set: ${PROJECT_NAME}_CXX_FLAGS
set(${PROJECT_NAME}_LINKER_FLAGS "-Wl,-rpath=${CMAKE_INSTALL_PREFIX}/lib,--enable-new-dtags ${ChimeraTK-DeviceAccess_LINKER_FLAGS}")

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
install(FILES ${PROJECT_BINARY_DIR}/${python_module} 
              DESTINATION ${install_path})
install(TARGETS ${boost_python_core_module} 
              LIBRARY DESTINATION ${install_path})

