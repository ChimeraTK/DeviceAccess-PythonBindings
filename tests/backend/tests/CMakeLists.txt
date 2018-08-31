list(APPEND list_of_sources 
            testDBase.cc
            testDBaseElem.cc
            test_new
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
                        ${mtca4u-deviceaccess_LIBRARIES}
                COMPILE_OPTIONS 
                        ${mtca4u-deviceaccess_CXX_FLAGS})
