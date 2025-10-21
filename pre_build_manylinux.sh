#!/usr/bin/env bash

# For manylinux_2_28

# Dependencies versions
export CPPEXT_VERSION="01.07.00"
export EXPRTK_VERSION="01.04.02"
export NLOHMANN_JSON_VERSION="v3.12.0"
export DEVICEACCESS_VERSION="03.24.01"

export PROJECT_PWD=`pwd`
export PROCS=`nproc`

# Creating local build directories
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}

# Helper function to build cmake-based projects
build_cmake () {
    cd $1
    mkdir builddir
    cd builddir/
    cmake -DCMAKE_BUILD_TYPE=Release -DJSON_BuildTests=OFF -DBUILD_TESTING=OFF -DBUILD_TESTS=OFF ..
    make install -j${PROCS}
    cd ${BUILD_DIR}
}

# Install required system packages
dnf install -y epel-release
dnf -y install gcc-c++ gcc-toolset-14-libatomic-devel libxml++-devel boost1.78-devel

g++ --version

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

cd ${PROJECT_PWD}

