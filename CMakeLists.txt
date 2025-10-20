cmake_minimum_required( VERSION 3.16 )
project( ChimeraTK-DeviceAccess-PythonBindings )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules )

set(${PROJECT_NAME}_MAJOR_VERSION 04)
set(${PROJECT_NAME}_MINOR_VERSION 00)
set(${PROJECT_NAME}_PATCH_VERSION 01)
include(cmake/set_version_numbers.cmake)

# set default compiler flags (C++)
include( cmake/set_default_build_to_release.cmake )
include( cmake/set_default_flags.cmake )
include( cmake/enable_code_style_check.cmake )
include( cmake/enable_code_coverage_report.cmake )
include( cmake/add_linter_target.cmake )

# ==============================================================================#

# Find dependencies
find_package(ChimeraTK-DeviceAccess 03.23 REQUIRED)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 2.10)

find_program(STUBGEN stubgen)

if(${STUBGEN} STREQUAL "STUBGEN-NOTFOUND")
  message(FATAL_ERROR "Could not find program 'stubgen'. Try installing package 'mypy' or similar.")
endif()

# ==============================================================================#

# Define library for the C++ part
include_directories(${CMAKE_SOURCE_DIR}/include)

file( GLOB_RECURSE headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*" )
aux_source_directory( "${CMAKE_CURRENT_SOURCE_DIR}/src/" sources )

pybind11_add_module( ${PROJECT_NAME} ${headers} ${sources} )

target_link_libraries( ${PROJECT_NAME} PRIVATE ChimeraTK::ChimeraTK-DeviceAccess )

set_property( TARGET ${PROJECT_NAME} PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE )

# set the right output file name for the the python core module library
set_target_properties( ${PROJECT_NAME} PROPERTIES PREFIX "" OUTPUT_NAME "deviceaccess" )

# Copy Python part to build directory (required for tests to run)
file( COPY mtca4u.py DESTINATION ${PROJECT_BINARY_DIR} )

# ==============================================================================#
# Python module stub
if((NOT ${CMAKE_BUILD_TYPE} STREQUAL "asan") AND(NOT ${CMAKE_BUILD_TYPE} STREQUAL "tsan"))
  configure_file(cmake/runStubgen.sh.in ${CMAKE_CURRENT_BINARY_DIR}/runStubgen.sh)
  add_custom_target(${PROJECT_NAME}_interface ALL
    COMMAND ./runStubgen.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${PROJECT_NAME}
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pyi
    COMMENT "Use stubgen to create .pyi for statement completion")
else()
  # stubgen would fail to run due to missing asan/tsan symbols
  message(NOTICE "Note: Python stub will not be generated for asan and tsan builds.")
endif()

# ==============================================================================#
ENABLE_TESTING()

# Copy the test scripts and test configuration files to the build directory:
file( COPY ${CMAKE_SOURCE_DIR}/tests DESTINATION ${PROJECT_BINARY_DIR} )

# Add the scripts to ctest
FILE( GLOB test_scripts "${PROJECT_BINARY_DIR}/tests/*.py" )

foreach( script_path ${test_scripts} )
  get_filename_component( test_name ${script_path} NAME_WE )

  if( NOT "${CMAKE_BUILD_TYPE}" STREQUAL "asan" AND NOT "${CMAKE_BUILD_TYPE}" STREQUAL "tsan" )
    add_test( NAME ${test_name} COMMAND ${Python_EXECUTABLE} ${script_path} WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/tests )
    set_tests_properties( ${test_name} PROPERTIES TIMEOUT 30 )
  endif()
endforeach( script_path )

# ==============================================================================#

# add target for documentation if sphinx is available
find_package( Sphinx )

if( SPHINX_EXECUTABLE )
  configure_file( "${PROJECT_SOURCE_DIR}/doc/conf.py.in" "${PROJECT_BINARY_DIR}/conf.py" )

  # copy the config file to the build directory
  message( STATUS "Html documentation support enabled; use 'make doc' to build it." )
  add_custom_target( doc ALL
    COMMAND "${SPHINX_EXECUTABLE}" -c "${PROJECT_BINARY_DIR}" -b html "${PROJECT_SOURCE_DIR}/doc" "doc/html"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating HTML documentation" VERBATIM )

  # unfortunately sphinx needs to scan the library, so we first have to run the build step before we get proper
  # documentation
  add_dependencies( doc ${PROJECT_NAME} )

  install( DIRECTORY ${CMAKE_BINARY_DIR}/doc/ DESTINATION share/doc/${PROJECT_NAME}-${${PROJECT_NAME}_SOVERSION}
    COMPONENT doc OPTIONAL )

  set( DOC_TARGET_ADDED TRUE CACHE INTERNAL "Doc target has been configured" )
else()
  message( STATUS "HTML doumentation support will not be enabled" )
endif()

# ==============================================================================#

if(SKBUILD)
  install( FILES ${PROJECT_BINARY_DIR}/deviceaccess.pyi 
           DESTINATION "${PROJECT_SOURCE_DIR}/src/deviceaccess")
  install( FILES ${PROJECT_SOURCE_DIR}/mtca4u.py 
           DESTINATION "${PROJECT_SOURCE_DIR}/src/mtca4u"
           RENAME __init__.py)
  install( TARGETS ${PROJECT_NAME} 
           LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/src/deviceaccess"
           RENAME deviceaccess.so)
else()
  # install Python modules to correct platform-dependent directory (if installing to system prefix)
  if( "${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr" OR "${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr/local" )
    set( install_path ${Python_SITEARCH} )
  else()
    set( install_path "lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages" )
  endif()
  
  install( FILES ${PROJECT_BINARY_DIR}/deviceaccess.pyi DESTINATION ${install_path} )
  install( FILES ${PROJECT_SOURCE_DIR}/mtca4u.py DESTINATION ${install_path} )
  install( TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${install_path} )
endif()
