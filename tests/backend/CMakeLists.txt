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
find_package(mtca4u-deviceaccess 
  ${min_req_mtca4u-deviceaccess_version} REQUIRED)
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
separate_arguments(mtca4u-deviceaccess_CXX_FLAGS)

#
# remove the -std=c++11 flag from mtca4u-deviceaccess_CXX_FLAGS
# The project requires c++14.
##############################################################################
list(REMOVE_ITEM mtca4u-deviceaccess_CXX_FLAGS -std=c++11)


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
      backend_registration.cc
      register_access.cc
      register_list.cc)
      
set(source_headers
      backend.h
      register_access.h
      test_catalogue.h
      register_list.h)

#
###############################################################################
add_library(${library_name} SHARED "")
target_sources(test_backend
  PRIVATE
    ${source_files}
    ${source_headers})
    
target_include_directories(${library_name}
  PRIVATE
    ${mtca4u-deviceaccess_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS})

target_link_libraries(test_backend
  PRIVATE
    ${mtca4u-deviceaccess_LIBRARIES} 
    ${Boost_LIBRARIES})

target_compile_options(${library_name}
  PRIVATE
      ${mtca4u-deviceaccess_CXX_FLAGS}
      -Wall
      -Weffc++
      -Wpedantic
      -Wextra
  PUBLIC
      -std=c++14)
      
