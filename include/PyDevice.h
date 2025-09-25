// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "PyOneDRegisterAccessor.h"
#include "PyScalarRegisterAccessor.h"
#include "PyTwoDRegisterAccessor.h"
#include "PyVoidRegisterAccessor.h"

#include <ChimeraTK/Device.h>
#include <ChimeraTK/SupportedUserTypes.h>
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
        const std::string& registerPathName, const py::list& accessModeFlags = py::list());
    PyScalarRegisterAccessor getScalarRegisterAccessor(const py::object& dType, const std::string& registerPathName,
        int elementsOffset = 0, const py::list& accessModeFlags = py::list());
    PyOneDRegisterAccessor getOneDRegisterAccessor(const py::object& dType, const std::string& registerPathName,
        int numberOfElements = 0, int elementsOffset = 0, const py::list& accessModeFlags = py::list());
    PyTwoDRegisterAccessor getTwoDRegisterAccessor(const py::object& dType, const std::string& registerPathName,
        int numberOfElements = 0, int elementsOffset = 0, const py::list& accessModeFlags = py::list());

    void activateAsyncRead();

    ChimeraTK::RegisterCatalogue getRegisterCatalogue();
    std::string getCatalogueMetadata(const std::string& parameterName);

    void writeArray(const std::string& registerPath, py::object& data, const py::object& dtype,
        size_t numberOfWords = 0, size_t wordOffsetInRegister = 0, const py::list& flaglist = py::list());

    void writeArray(const std::string& registerPath, py::array& data, const py::object& dtype, size_t numberOfWords = 0,
        size_t wordOffsetInRegister = 0, const py::list& flaglist = py::list());
    void writeList(const std::string& registerPath, py::list& data, const py::object& dtype, size_t numberOfWords = 0,
        size_t wordOffsetInRegister = 0, const py::list& flaglist = py::list());
    void writeScalar(const std::string& registerPath, UserTypeVariantNoVoid& data, const py::object& dtype,
        size_t numberOfWords = 0, size_t wordOffsetInRegister = 0, const py::list& flaglist = py::list());

    pybind11::array read(const std::string& registerPath, const py::object& dtype, size_t numberOfElements = 0,
        size_t elementsOffset = 0, const py::list& flaglist = py::list());

    static void bind(py::module& mod);

   private:
    Device _device;
  };

} // namespace ChimeraTK
