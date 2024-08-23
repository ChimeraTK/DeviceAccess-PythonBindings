#!/usr/bin/env bash

# For manylinux_2_28

export PROJECT_PWD=`pwd`
export PROCS=`nproc`

# Creating local build directories
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
mkdir ${BUILD_DIR}/local
mkdir ${BUILD_DIR}/local/bin
mkdir ${BUILD_DIR}/local/include
mkdir ${BUILD_DIR}/local/lib

# Helper function to build meson-based projects
build_ninja () {
  cd $1
  meson setup build --buildtype=release --prefix=${BUILD_DIR}/local/ --libdir=lib --includedir=lib/include
  ninja -j${PROCS} -C build
  ninja install -C build
  cd ${BUILD_DIR}
}

# Helper function to build cmake-based projects
build_cmake () {
  cd $1
  mkdir builddir
  cd builddir/
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make -j${PROCS}
  make install
  cd ${BUILD_DIR}
}

# Install required system packages
dnf install -y epel-release
dnf update -y
dnf upgrade -y
dnf groupinstall -y "Development Tools"
dnf -y install clang zeromq-devel readline-devel opencv-devel openldap-devel wget libssh2-devel boost boost-devel libxml++-devel doxygen patchelf perl git libmodbus-devel rsync cmake gcc-toolset-12-libatomic-devel

pipx install meson ninja sphinx

# Dependencies versions
export YAJL_VERSION="2.1.0"
export EPICS_VERSION="R7.0.8"
export CHIMERATK_EPICS_VERSION="01.00.00"
export OPEN62541_VERSION="v1.3.1"
export LIBTIRPC_VERSION="1.3.4"
export TINE_VERSION="08ca83228932c3b08caf3b4c6578f1da8e65f5c5"
export DOOCS_VERSION="DOOCSVERSION_24_3_1"
export DOOCS_SERVER_TEST_HELPER_VERSION="01.07.00"
export CPPEXT_VERSION="01.05.02"
export EXPRTK_VERSION="01.04.01"
export NLOHMANN_JSON_VERSION="v3.7.3"
export DEVICEACCESS_VERSION="03.15.02"
export DEVICEACCESS_EPICS_VERSION="01.00.00"
export DEVICEACCESS_OPCUA_VERSION="01.03.00"
export DEVICEACCESS_DOOCS_VERSION="01.09.01"
export DEVICEACCESS_MODBUS_VERSION="01.05.00"

# Install EPICS
mkdir ${BUILD_DIR}/EPICS
cd ${BUILD_DIR}/EPICS
git clone --recursive --depth 1 --branch ${EPICS_VERSION} https://github.com/epics-base/epics-base.git
cd epics-base/
make -j${PROCS}
export EPICS_BASE=${BUILD_DIR}/EPICS/epics-base
export EPICS_HOST_ARCH=$(${EPICS_BASE}/startup/EpicsHostArch)
export PATH=${EPICS_BASE}/bin/${EPICS_HOST_ARCH}:${PATH}
cd ${BUILD_DIR}

# Install libtirpc-1.3
wget https://downloads.sourceforge.net/libtirpc/libtirpc-${LIBTIRPC_VERSION}.tar.bz2
tar -xvf libtirpc-${LIBTIRPC_VERSION}.tar.bz2
cd libtirpc-${LIBTIRPC_VERSION}/
./configure --prefix=${BUILD_DIR}/local
make -j${PROCS}
make install
cd ${BUILD_DIR}

# Install TINE
git clone --recursive http://doocs-git.desy.de/cgit/vendor/desy/mcs/tine/tine-package.git
cd tine-package/
git checkout ${TINE_VERSION}
cd doocs/
./prepare
cd build.LINUX/
make -j${PROCS}
make install

# Ugly workaround for cp errors
cp ./src/*.so /usr/local/lib/
cd ${BUILD_DIR}

# Install GUL
git clone --recursive --depth 1 --branch ${DOOCS_VERSION} https://mcs-gitlab.desy.de/doocs/doocs-core-libraries/gul.git
build_ninja gul

# Install DOOCS clientlib
export ENSHOST=ldap://xfelens1.desy.de
git clone --recursive --depth 1 --branch ${DOOCS_VERSION} https://mcs-gitlab.desy.de/doocs/doocs-core-libraries/clientlib.git
build_ninja clientlib

# Install DOOCS serverlib
git clone --recursive --depth 1 --branch ${DOOCS_VERSION} https://mcs-gitlab.desy.de/doocs/doocs-core-libraries/serverlib.git
build_ninja serverlib

# Install DOOCSServerTestHelper
git clone --recursive --depth 1 --branch ${DOOCS_SERVER_TEST_HELPER_VERSION} https://github.com/ChimeraTK/DoocsServerTestHelper.git
build_cmake DoocsServerTestHelper

#Install ChimeraTK-EPICS
git clone --recursive --depth 1 --branch ${CHIMERATK_EPICS_VERSION} https://github.com/ChimeraTK/EPICS-Interface.git
cd EPICS-Interface/
mkdir builddir
cd builddir/
cmake -DCMAKE_BUILD_TYPE=Release -DEPICS_VERSION=7 ..
make -j${PROCS}
make install
cd ${BUILD_DIR}

# Install yajl
git clone --recursive --depth 1 --branch ${YAJL_VERSION} https://github.com/lloyd/yajl.git
build_cmake yajl

#Install open62541
git clone --recursive --depth 1 --branch ${OPEN62541_VERSION} https://github.com/open62541/open62541.git
build_cmake open62541

# Install cppext
git clone --recursive --depth 1 --branch ${CPPEXT_VERSION} https://github.com/ChimeraTK/cppext.git
build_cmake cppext

# Install exprtk
git clone --recursive --depth 1 --branch ${EXPRTK_VERSION} https://github.com/ChimeraTK/exprtk-interface.git
build_cmake exprtk-interface

# Install nlohmann-json
git clone --recursive --depth 1 --branch ${NLOHMANN_JSON_VERSION} https://github.com/nlohmann/json.git
build_cmake json

# Install ChimeraTK-DeviceAccess
git clone --recursive --depth 1 --branch ${DEVICEACCESS_VERSION} https://github.com/ChimeraTK/DeviceAccess.git
build_cmake DeviceAccess

# Install ChimeraTK-DeviceAccess-EpicsBackend
git clone --recursive --depth 1 --branch ${DEVICEACCESS_EPICS_VERSION} https://github.com/ChimeraTK/DeviceAccess-EpicsBackend.git
build_cmake DeviceAccess-EpicsBackend

# Install ChimeraTK-DeviceAccess-OpcUaBackend
git clone --recursive --depth 1 --branch ${DEVICEACCESS_OPCUA_VERSION} https://github.com/ChimeraTK/DeviceAccess-OpcUaBackend.git
cd DeviceAccess-OpcUaBackend
mkdir builddir
cd builddir/
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j${PROCS} ChimeraTK-DeviceAccess-OPC-UA-Backend
make install/fast
cd ${BUILD_DIR}

# Install ChimeraTK-DeviceAccess-DoocsBackend
git clone --recursive --depth 1 --branch ${DEVICEACCESS_DOOCS_VERSION} https://github.com/ChimeraTK/DeviceAccess-DoocsBackend.git
build_cmake DeviceAccess-DoocsBackend

# Install ChimeraTK-DeviceAccess-ModbusBackend
git clone --recursive --depth 1 --branch ${DEVICEACCESS_MODBUS_VERSION} https://github.com/ChimeraTK/DeviceAccess-ModbusBackend.git
build_cmake DeviceAccess-ModbusBackend

cd ${PROJECT_PWD}

