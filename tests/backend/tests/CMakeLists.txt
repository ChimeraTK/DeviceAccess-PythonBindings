list(APPEND list_of_sources 
            testDBase.cc
            testDBaseElem.cc
            testRegisterList.cc)
            
# usage: register_tests(SOURCES list_of_source_files 
#                       NAMESPACE test_namespace 
#                       DEPENDS list_of_targets
#                       INCLUDES list_of_include_directories
#                       COMPILE_OPTIONS list_of_compile_options)
# function defined in registerTests.cmake
register_tests( SOURCES ${list_of_sources}
                NAMESPACE "TestBackend" 
                INCLUDES ${CMAKE_CURRENT_LIST_DIR} 
                         ${CMAKE_SOURCE_DIR}
                DEPENDS ${library_name}
                        ${mtca4u-deviceaccess_LIBRARIES}
                COMPILE_OPTIONS 
                        ${mtca4u-deviceaccess_CXX_FLAGS})
