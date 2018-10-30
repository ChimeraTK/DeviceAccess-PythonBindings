include(registerTests)

list(APPEND list_of_sources
            test_Register.cc
            testRegisterList.cc)
            
get_filename_component(backend_source_dir ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)

# usage: register_tests(SOURCES list_of_source_files 
#                       NAMESPACE test_namespace 
#                       DEPENDS list_of_targets
#                       INCLUDES list_of_include_directories
#                       COMPILE_OPTIONS list_of_compile_options)
# function defined in registerTests.cmake
register_tests( SOURCES ${list_of_sources}
                NAMESPACE "TestBackend" 
                INCLUDES ${CMAKE_CURRENT_LIST_DIR} 
                         ${backend_source_dir}
                DEPENDS ${library_name}
                        ${ChimeraTK-DeviceAccess_LIBRARIES}
                COMPILE_OPTIONS 
                        ${ChimeraTK-DeviceAccess_CXX_FLAGS})
