cmake_minimum_required(VERSION 2.8)

project(MtcaMappedDevicePythonBindings)
set(python_module_name "mtcamappeddevice")

#
# Check project dependencies are met
find_package(Boost REQUIRED COMPONENTS python)
find_package(PythonLibs REQUIRED)
find_package(MtcaMappedDevice REQUIRED)


#
# This location has some of the macros that we are using below
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
include(helperMacros)

#
# Set the compiler flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -Wextra -Wshadow -pedantic -Wuninitialized")

#
# Set header file locations
include_directories(${Boost_INCLUDE_DIRS} 
                    ${PYTHON_INCLUDE_DIR}
                    ${MtcaMappedDevice_INCLUDE_DIRS})

#
# Define the build target for python module 
add_library(${python_module_name} SHARED MtcaMappedDevicePython.cc 
                                         devMapAdapter.cc
                                         devBaseAdapter.cc
                                         devBaseFactory.cc
                                         DeviceFactory.cc
                                         devMapFactory.cc
                                         HelperFunctions.cc
                                         RegisterAccessorWrapperFunctions.cc)
                                         
                                         
                                         
target_link_libraries(${python_module_name} ${MtcaMappedDevice_LIBRARIES}
                                            ${PYTHON_LIBRARIES} 
                                            ${Boost_LIBRARIES})
# donot prefix lib to the generated so:
set_target_properties(${python_module_name} PROPERTIES PREFIX "")


# Copy mymodule.py to the build directory. TODO: This is supposed to be a 
# temporary thing. Will have to structure into something better later
#
#TODO: Fix this hardcode later
configure_file( mtca4upy.py  ${PROJECT_BINARY_DIR} 
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
      CONFIGURE_FILE(cmake/Makefile.coverage
        ${PROJECT_BINARY_DIR}/Makefile.coverage @ONLY)
      ADD_CUSTOM_TARGET(coverage
        make -f Makefile.coverage
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating test coverage documentation" VERBATIM
      )
ENDIF(CMAKE_BUILD_TYPE STREQUAL "Debug")

