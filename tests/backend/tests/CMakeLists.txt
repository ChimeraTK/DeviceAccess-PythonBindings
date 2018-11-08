include(registerTests)

get_filename_component(backend_source_dir ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR} list_of_sources)

register_tests( SOURCES ${list_of_sources}
                NAMESPACE "TestBackend" 
                INCLUDE_DIRECTORIES
                         ${CMAKE_CURRENT_LIST_DIR}
                         ${ChimeraTK-DeviceAccess_INCLUDE_DIRS}
                         ${backend_source_dir}
                LINK_LIBRARIES
                        ${ChimeraTK-DeviceAccess_LINKER_FLAGS}
                        ${library_name}
                        ${ChimeraTK-DeviceAccess_LIBRARIES}
                COMPILE_OPTIONS 
                        ${ChimeraTK-DeviceAccess_CXX_FLAGS})


