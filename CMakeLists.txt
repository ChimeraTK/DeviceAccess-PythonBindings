
project(MtcaMappedDevicePythonBindings)
set(python_module_name "mtcamappeddevice")

# Check project dependencies are met
#
find_package(Boost REQUIRED COMPONENTS python)
find_package(PythonLibs REQUIRED)
find_package(MtcaMappedDevice REQUIRED)

# Set header file locations
#
include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIR}
    ${MtcaMappedDevice_INCLUDE_DIRS})

# Define the build target for python module 
#
add_library(${python_module_name} SHARED MtcaMappedDevicePython.cc)
target_link_libraries(${python_module_name} ${MtcaMappedDevice_LIBRARIES}
    ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
set_target_properties(${python_module_name} PROPERTIES PREFIX "") # so that the
                                                                  # generated
 # .so does not have the default lib prefix.

