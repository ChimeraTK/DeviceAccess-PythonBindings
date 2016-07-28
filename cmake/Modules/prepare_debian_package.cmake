
#Prepare the debian control files from the template.
#Basically this is setting the correct version number in most of the files


string(TOLOWER python-${PROJECT_NAME} PACKAGE_NAME)
set(PACKAGE_DEV_NAME "${PACKAGE_NAME}-dev")

#some variables needed for the make_debian_package.sh script

#we can use the package name here because it does not contain a hyphen
set(PACKAGE_BUILDVERSION_ENVIRONMENT_VARIABLE_NAME "${PROJECT_NAME}_BUILDVERSION")
#FIXME: This is redundant 
set(PACKAGE_BASE_NAME ${PACKAGE_NAME})
#FIXME: Do we need a full libary version incl. build-number?
set(PACKAGE_FULL_LIBRARY_VERSION ${${PROJECT_NAME}_VERSION})
set(PACKAGE_TAG_VERSION ${${PROJECT_NAME}_VERSION})
#FIXME: This is project specific, should not be in a module. Don't know how to solve it right now.
set(PACKAGE_GIT_URI "https://github.com/ChimeraTK/DeviceAccess-PythonBindings.git")
set(PACKAGE_MESSAGE "Debian package for MTCA4U deviceaccess python bindings ${${PROJECT_NAME}_VERSION}")

#Nothing to change, just copy
file(COPY ${CMAKE_SOURCE_DIR}/cmake/debian_package_templates/compat
           ${CMAKE_SOURCE_DIR}/cmake/debian_package_templates/rules
           ${CMAKE_SOURCE_DIR}/cmake/debian_package_templates/copyright
     DESTINATION debian_from_template)
file(COPY ${CMAKE_SOURCE_DIR}/cmake/debian_package_templates/source/format
     DESTINATION debian_from_template/source)

configure_file(${CMAKE_SOURCE_DIR}/cmake/debian_package_templates/control.in
               debian_from_template/control)
configure_file(${CMAKE_SOURCE_DIR}/cmake/debian_package_templates/package.install.in
               debian_from_template/${PACKAGE_NAME}.install)
configure_file(${CMAKE_SOURCE_DIR}/cmake/debian_package_templates/package-dev.install.in
               debian_from_template/${PACKAGE_NAME}-dev.install)
               
#Copy and configure the shell script which performs the actual
#building of the package
configure_file(${CMAKE_SOURCE_DIR}/cmake/make_debian_package.sh.in
               make_debian_package.sh @ONLY)
               
#A custom target so you can just run make debian_package
#(You could instead run make_debian_package.sh yourself, hm...)
add_custom_target(debian_package ${CMAKE_BINARY_DIR}/make_debian_package.sh
                  COMMENT Building debian package for tag ${mtca4u-deviceaccess_VERSION})

#For convenience: Also create an install script for DESY
set(PACKAGE_FILES_WILDCARDS "${PACKAGE_NAME}*.deb ${PACKAGE_NAME}*.changes")

configure_file(${CMAKE_SOURCE_DIR}/cmake/install_debian_package_at_DESY.sh.in
               install_debian_package_at_DESY.sh @ONLY)

