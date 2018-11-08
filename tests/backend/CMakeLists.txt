cmake_minimum_required(VERSION 3.5)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

option(TESTBACKEND_ENABLETESTING "enables tests if true" TRUE)

project(test_backend)

include(CTest)
set(library_name test_backend)


#
# Project dependencies:
# Use installed version of ChimeraTK-DeviceAccess-dev by default.
# When added as a subproject parent project can override
# min_req_chimeratk_version to force desired value.
###############################################################################
set(min_req_chimeratk_version "" CACHE STRING "")
find_package(ChimeraTK-DeviceAccess ${min_req_chimeratk_version} REQUIRED)
find_package(Boost COMPONENTS system REQUIRED)

#
# Workaround: ${ChimeraTK-DeviceAccess_CXX_FLAGS} is a string
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
# todo: the trailing white space problem has to be taken care on
# deviceaccess side.
string(STRIP ${ChimeraTK-DeviceAccess_LINKER_FLAGS}
             ChimeraTK-DeviceAccess_LINKER_FLAGS)

if (TESTBACKEND_ENABLETESTING)
  #
  # Register tests if user option is selected.
  ###########################################################################
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
    ${ChimeraTK-DeviceAccess_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS})

target_link_libraries(${library_name}
  PRIVATE
    ${ChimeraTK-DeviceAccess_LINKER_FLAGS}
    ${ChimeraTK-DeviceAccess_LIBRARIES}
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
      
#
###############################################################################
configure_file(cmake/TestBackend.dmap.in
    ${CMAKE_CURRENT_BINARY_DIR}/tests/TestBackend.dmap)
