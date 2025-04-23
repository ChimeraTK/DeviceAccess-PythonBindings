// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "PyOneDRegisterAccessor.h"
#include "PyScalarRegisterAccessor.h"
#include "PyTwoDRegisterAccessor.h"
#include "PyVoidRegisterAccessor.h"

#include <ChimeraTK/Device.h>
#include <ChimeraTK/VariantUserTypes.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

namespace ChimeraTK {

  class PyDevice {
   public:
    PyDevice() = default;
    ~PyDevice() = default;

    explicit PyDevice(const std::string& aliasName);

    void open(const std::string& aliasName);

    void open();
    void close();

    PyVoidRegisterAccessor getVoidRegisterAccessor(
        const std::string& registerPathName, const py::list& accessModeFlags);
    PyScalarRegisterAccessor getScalarRegisterAccessor(UserTypeVariantNoVoid& userType,
        const std::string& registerPathName, int elementsOffset, const py::list& accessModeFlags);
    PyOneDRegisterAccessor getOneDRegisterAccessor(UserTypeVariantNoVoid& userType, const std::string& registerPathName,
        int numberOfElements, int elementsOffset, const py::list& accessModeFlags);
    PyTwoDRegisterAccessor getTwoDRegisterAccessor(UserTypeVariantNoVoid& userType, const std::string& registerPathName,
        int numberOfElements, int elementsOffset, const py::list& accessModeFlags);

    void activateAsyncRead();

    static void bind(py::module& mod);

   private:
    Device _device;
  };

} // namespace ChimeraTK
