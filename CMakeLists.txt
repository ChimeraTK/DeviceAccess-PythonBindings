cmake_minimum_required(VERSION 2.8)

project(mtca4uPy)
#==============================================================================#
# This is a hack to get escape ${} as the variable expansion in the files copied
# through configure_file. This is handy when copying shell scripts.
# TODO: move this to a better location
set(Dollar "$")
#==============================================================================#
# version Information for the project
# bump up the version numbers in this section when creating new tags 
set(${PROJECT_NAME}_MAJOR_VERSION 00)
set(${PROJECT_NAME}_MINOR_VERSION 01)
set(${PROJECT_NAME}_PATCH_VERSION 00)
set(min_req_mtcamappeddev_version 00.17)

set(${PROJECT_NAME}_VERSION
  ${${PROJECT_NAME}_MAJOR_VERSION}.${${PROJECT_NAME}_MINOR_VERSION}.${${PROJECT_NAME}_PATCH_VERSION})
set(${PROJECT_NAME}_SOVERSION ${${PROJECT_NAME}_MAJOR_VERSION}.${${PROJECT_NAME}_MINOR_VERSION})

# other variables 
set(boost_python_core_module "mtcamappeddevice")
set(python_module "mtca4u.py")
set(sphinx_build_confg_in "${PROJECT_SOURCE_DIR}/cmake/conf.py.in")
set(location_of_rst_source_files "${PROJECT_SOURCE_DIR}/doc")
set(location_of_built_html_files "${PROJECT_BINARY_DIR}/html")
set(sphinx_build_config_file "${PROJECT_BINARY_DIR}/conf.py")
set(author "MSK")
#
# This location has some of the modules that we are using below
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
include(helperMacros)

#
# Check if project dependencies are met
find_package(Boost REQUIRED COMPONENTS python)
find_package(PythonLibs REQUIRED)
find_package(MtcaMappedDevice ${min_req_mtcamappeddev_version} REQUIRED)
find_package(NumPy REQUIRED)

#
# add target for documentation if sphinx v1.3+ is available. 
CHECK_FOR_SPHINX_BUILD()
if(SUPPORTED_SPHINX_BUILD_VERSION)
  ADD_HTML_DOCUMENTATION_SUPPORT()
else()
    message(STATUS "Html doumentation support will not be enabled")
    message(STATUS "Please install working sphinx-doc version (1.3.0+) for html documentation support")
endif(SUPPORTED_SPHINX_BUILD_VERSION)



#
# Set the compiler flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -Weffc++ -Wextra -Wshadow -pedantic -Wuninitialized")

#
# Set header file locations
include_directories(SYSTEM ${Boost_INCLUDE_DIRS} 
                            ${PYTHON_INCLUDE_DIR}
                            ${MtcaMappedDevice_INCLUDE_DIRS})

#
# Define the build target for python module 
add_library(${boost_python_core_module} SHARED MtcaMappedDevicePython.cc 
                                         devMapAdapter.cc
                                         devBaseAdapter.cc
                                         devBaseFactory.cc
                                         DeviceFactory.cc
                                         devMapFactory.cc
                                         HelperFunctions.cc
                                         RegisterAccessorWrapperFunctions.cc)
                                         
                                         
                                         
target_link_libraries(${boost_python_core_module} ${MtcaMappedDevice_LIBRARIES}
                                            ${PYTHON_LIBRARIES} 
                                            ${Boost_LIBRARIES})
# donot prefix lib to the generated so:
set_target_properties(${boost_python_core_module} PROPERTIES PREFIX "")


# Copy mymodule.py to the build directory. TODO: This is supposed to be a 
# temporary thing. Will have to structure into something better later
#
#TODO: Fix this hardcode later
configure_file( ${python_module}  ${PROJECT_BINARY_DIR} 
                copyonly)

#
# For now tests and coverage are enabled in the debug Mode only
IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
    ENABLE_TESTING()
    #
    # Enable gdb and coverage report in the debug configuration.
    # (-DCMAKE_BUILD_TYPE=Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 --coverage")
    #
    # Copy the test scripts to the build directory:
    COPY_CONTENT_TO_BUILD_DIR("tests;tests/mapfiles")

    #
    # Add the scripts as ctests.
    FILE(GLOB location_of_script_files 
        "${PROJECT_BINARY_DIR}/tests/*.py")
    ADD_SCRIPTS_AS_TESTS( "${location_of_script_files}" )

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


# Install the shared obhect and the python module in "proper" places?
install(TARGETS ${boost_python_core_module}
        LIBRARY DESTINATION usr/lib/)
install(FILES ${python_module}
        DESTINATION usr/lib/)


# cmake instructions for packaging:
include(${CMAKE_SOURCE_DIR}/cmake/prepare_debian_package.cmake)
