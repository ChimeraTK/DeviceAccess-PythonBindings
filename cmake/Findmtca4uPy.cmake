#
# cmake module for finding mtca4uPy
#
# returns:
#   mtca4uPy_FOUND        : true or false, depending on whether
#                           the package was found
#   mtca4uPy_VERSION      : the package version
#   mtca4uPy_INCLUDE_DIRS : path to the include directory
#   mtca4uPy_LIBRARY_DIRS : path to the library directory
#   mtca4uPy_LIBRARY      : the provided libraries
#
# @author Martin Killenberg, DESY (modified for mtca4uPy by Martin Hierholzer, DESY)
#

SET(mtca4uPy_FOUND 0)

#FIXME: the search path for the device config has to be extended/generalised/improved
FIND_PATH(mtca4uPy_DIR
    mtca4uPyConfig.cmake
    ${CMAKE_CURRENT_LIST_DIR}
    )

#Once we have found the config our job is done. Just load the config which provides the required 
#varaibles.
include(${mtca4uPy_DIR}/mtca4uPyConfig.cmake)

#use a macro provided by CMake to check if all the listed arguments are valid
#and set mtca4uPy_FOUND accordingly
include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(mtca4uPy 
        REQUIRED_VARS mtca4uPy_LIBRARIES mtca4uPy_INCLUDE_DIRS
	VERSION_VAR mtca4uPy_VERSION )

