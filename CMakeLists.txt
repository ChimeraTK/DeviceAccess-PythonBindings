cmake_minimum_required(VERSION 3.16)
project(ChimeraTK-DeviceAccess-PythonBindings)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

set(${PROJECT_NAME}_MAJOR_VERSION 03)
set(${PROJECT_NAME}_MINOR_VERSION 06)
set(${PROJECT_NAME}_PATCH_VERSION 00)
include(cmake/set_version_numbers.cmake)

# set default compiler flags (C++)
include(cmake/set_default_build_to_release.cmake)
include(cmake/set_default_flags.cmake)

include(cmake/enable_code_style_check.cmake)

# ==============================================================================#

# Find dependencies
find_package(ChimeraTK-DeviceAccess 03.23 REQUIRED)

find_package(Python 3 REQUIRED COMPONENTS Interpreter Development NumPy)

# Convert Python version into the format we need for BOOST component names
set(python_version_no_dot "${Python_VERSION_MAJOR}${Python_VERSION_MINOR}")

find_package(Boost REQUIRED COMPONENTS python${python_version_no_dot} numpy${python_version_no_dot})

# ==============================================================================#

# Defining NPY_NO_DEPRECATED_API as NPY_1_7_API_VERSION, should ensure that the
# compilation does not go through if the code uses API's marked as deprecated in
# numpy v1.7's documentation.
#
# Reference:
# https://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html
add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

# ==============================================================================#

# Define library for the C++ part
include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB_RECURSE headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*")
file(GLOB_RECURSE sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*")

add_library(${PROJECT_NAME} SHARED ${headers} ${sources})

target_link_libraries(${PROJECT_NAME}
  PRIVATE ChimeraTK::ChimeraTK-DeviceAccess
  PRIVATE Python::NumPy
  PRIVATE Python::Module
  PRIVATE ${Boost_LIBRARIES}
  PRIVATE Boost::python${python_version_no_dot}
  PRIVATE Boost::numpy${python_version_no_dot})

# do not remove runtime path to libmtca-deviceaccess location from ${boost_python_core_module} when installing
set_property(TARGET ${PROJECT_NAME} PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

# set the right output file name for the the python core module library
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" OUTPUT_NAME "_da_python_bindings")

# Copy Python part to build directory (required for tests to run)
file(COPY mtca4u.py DESTINATION ${PROJECT_BINARY_DIR})
file(COPY deviceaccess.py DESTINATION ${PROJECT_BINARY_DIR})

# ==============================================================================#
ENABLE_TESTING()

# Copy the test scripts and test configuration files to the build directory:
file(COPY ${CMAKE_SOURCE_DIR}/tests DESTINATION ${PROJECT_BINARY_DIR})

# Add the scripts to ctest
FILE(GLOB test_scripts "${PROJECT_BINARY_DIR}/tests/*.py")

foreach(script_path ${test_scripts})
  get_filename_component(test_name ${script_path} NAME_WE)

  if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "asan" AND NOT "${CMAKE_BUILD_TYPE}" STREQUAL "tsan")
    add_test(NAME ${test_name} COMMAND ${Python_EXECUTABLE} ${script_path} WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/tests)
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 30)
  endif()
endforeach(script_path)

include(cmake/enable_code_coverage_report.cmake)

# ==============================================================================#

# add target for documentation if sphinx is available
find_package(Sphinx)

if(SPHINX_EXECUTABLE)
  configure_file("${PROJECT_SOURCE_DIR}/doc/conf.py.in" "${PROJECT_BINARY_DIR}/conf.py")

  # copy the config file to the build directory
  message(STATUS "Html documentation support enabled; use 'make doc' to build it.")
  add_custom_target(doc ALL
    COMMAND "${SPHINX_EXECUTABLE}" -c "${PROJECT_BINARY_DIR}" -b html "${PROJECT_SOURCE_DIR}/doc" "doc/html"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating HTML documentation" VERBATIM)

  # unfortunately sphinx needs to scan the library, so we first have to run the build step before we get proper
  # documentation
  add_dependencies(doc ${PROJECT_NAME})

  install(DIRECTORY ${CMAKE_BINARY_DIR}/doc/ DESTINATION share/doc/${PROJECT_NAME}-${${PROJECT_NAME}_SOVERSION}
    COMPONENT doc OPTIONAL)

  set(DOC_TARGET_ADDED TRUE CACHE INTERNAL "Doc target has been configured")
else()
  message(STATUS "HTML doumentation support will not be enabled")
endif()

# ==============================================================================#

# install Python modules to correct platform-dependent directory (if installing to system prefix)
if("${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr" OR "${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr/local")
  set(install_path ${Python_SITEARCH})
else()
  set(install_path "lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
endif()

install(FILES ${PROJECT_SOURCE_DIR}/deviceaccess.py DESTINATION ${install_path})
install(FILES ${PROJECT_SOURCE_DIR}/mtca4u.py DESTINATION ${install_path})
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${install_path})
