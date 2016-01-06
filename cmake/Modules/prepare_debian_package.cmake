
#Prepare the debian control files from the template.
#Basically this is setting the correct version number in most of the files


string(TOLOWER python-${PROJECT_NAME} PACKAGE_NAME)
set(PACKAGE_DEV_NAME "${PACKAGE_NAME}-dev")

#Nothing to change, just copy
file(COPY ${CMAKE_SOURCE_DIR}/cmake/debian_package_templates/compat
           ${CMAKE_SOURCE_DIR}/cmake/debian_package_templates/rules
     DESTINATION debian_from_template)
file(COPY ${CMAKE_SOURCE_DIR}/cmake/debian_package_templates/source/format
     DESTINATION debian_from_template/source)

configure_file(${CMAKE_SOURCE_DIR}/cmake/debian_package_templates/control.in
               debian_from_template/control)
configure_file(${CMAKE_SOURCE_DIR}/cmake/debian_package_templates/package.install.in
               debian_from_template/${PACKAGE_NAME}.install)
configure_file(${CMAKE_SOURCE_DIR}/cmake/debian_package_templates/package-dev.install.in
               debian_from_template/${PACKAGE_NAME}-dev.install)
               

#Set the version number
configure_file(${CMAKE_SOURCE_DIR}/cmake/debian_package_templates/copyright.in
               debian_from_template/copyright @ONLY)

#Copy and configure the shell script which performs the actual
#building of the package
configure_file(${CMAKE_SOURCE_DIR}/cmake/make_debian_package.sh.in
               make_debian_package.sh)
               
#A custom target so you can just run make debian_package
#(You could instead run make_debian_package.sh yourself, hm...)
add_custom_target(debian_package ${CMAKE_BINARY_DIR}/make_debian_package.sh
                  COMMENT Building debian package for tag ${mtca4u-deviceaccess_VERSION})

#For convenience: Also create an install script for DESY
set(PACKAGE_FILES_WILDCARDS "${PACKAGE_NAME}*.deb ${PACKAGE_NAME}*.changes")

configure_file(${CMAKE_SOURCE_DIR}/cmake/install_debian_package_at_DESY.sh.in
               install_debian_package_at_DESY.sh @ONLY)

