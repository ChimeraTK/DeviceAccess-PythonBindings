cmake_minimum_required(VERSION 3.5)
project(test_backend)

option(ENABLE_TESTING "enables testing if true" TRUE)
set(library_name test_backend)


#
# Project dependencies:
# Use installed version of libmtca4u-deviceaccess-dev by default.
# But when added as a subproject min_req_mtca4u-deviceaccess_version may
# be overridden from the parent project
###############################################################################
set(min_req_mtca4u-deviceaccess_version "" CACHE STRING "")
find_package(ChimeraTK-DeviceAccess ${min_req_chimeratk_version} REQUIRED)
find_package(Boost COMPONENTS system REQUIRED)

#
# Workaround: ${mtca4u-deviceaccess_CXX_FLAGS} is a string
# "-DFUSION_MAX_MAP_SIZE=30 -DFUSION_MAX_VECTOR_SIZE=30 -std=c++11" 
#
# because target_compile options works on a list passing the string above causes
# it to dump all charachters including  " as the compiler options; this dosenot
# work for gcc.
#
# Would need to convert the string to a list before using with
# target_compile_options. seperate_arguments can do this for us.
##############################################################################
separate_arguments(ChimeraTK-DeviceAccess_CXX_FLAGS)

if (ENABLE_TESTING)
  #
  # Add support for custom functions for registering test targets.
  ###########################################################################
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
  include(registerTests)
  enable_testing()
  add_subdirectory(tests)
endif()

#
###############################################################################
set(source_files
      backend.cc
      register.cc
      backend_registration.cc
      register_list.cc)
      
set(source_headers
      backend.h
      backend_registration.h
      test_catalogue.h
      register.h
      variant_types.h
      register_list.h)

#
###############################################################################
add_library(${library_name} SHARED "")
target_sources(${library_name}
  PRIVATE
    ${source_files}
    ${source_headers})
    
target_include_directories(${library_name}
  SYSTEM
  PRIVATE
    ${mtca4u-deviceaccess_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS})

target_link_libraries(${library_name}
  PRIVATE
    ${mtca4u-deviceaccess_LIBRARIES} 
    ${Boost_LIBRARIES})

target_compile_options(${library_name}
  PRIVATE
     ${ChimeraTK-DeviceAccess_CXX_FLAGS}
      -Wall
      -Weffc++
      -Wpedantic
      -Wextra
  PUBLIC
      -std=c++17)
      
configure_file(cmake/TestBackend.dmap.in ${CMAKE_CURRENT_BINARY_DIR}/tests/TestBackend.dmap)
