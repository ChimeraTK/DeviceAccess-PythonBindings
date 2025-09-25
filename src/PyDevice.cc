// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyDevice.h"

#include "HelperFunctions.h"
#include "PyOneDRegisterAccessor.h"
#include "PyScalarRegisterAccessor.h"
#include "PyVoidRegisterAccessor.h"

#include <ChimeraTK/AccessMode.h>
#include <ChimeraTK/NDRegisterAccessor.h>
#include <ChimeraTK/SupportedUserTypes.h>
#include <ChimeraTK/VariantUserTypes.h>
#include <ChimeraTK/VoidRegisterAccessor.h>

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <boost/smart_ptr/shared_ptr.hpp>

namespace py = pybind11;

namespace ChimeraTK {

  // Helper function to convert Python list of flags to ChimeraTK::AccessModeFlags
  ChimeraTK::AccessModeFlags convertFlagsFromPython(const py::list& flaglist) {
    ChimeraTK::AccessModeFlags flags{};
    for(auto flag : flaglist) {
      flags.add(flag.cast<ChimeraTK::AccessMode>());
    }
    return flags;
  }

  PyDevice::PyDevice(const std::string& aliasName) {
    _device = Device(aliasName);
  }

  void PyDevice::open(const std::string& aliasName) {
    _device.open(aliasName);
  }
  void PyDevice::open() {
    _device.open();
  }

  void PyDevice::close() {
    _device.close();
  }

  PyVoidRegisterAccessor PyDevice::getVoidRegisterAccessor(
      const std::string& registerPathName, const py::list& accessModeFlags) {
    auto acc = _device.getVoidRegisterAccessor(registerPathName, convertFlagsFromPython(accessModeFlags));
    return PyVoidRegisterAccessor{acc.getImpl()};
  }

  PyScalarRegisterAccessor PyDevice::getScalarRegisterAccessor(const py::object& dType,
      const std::string& registerPathName, int elementsOffset, const py::list& accessModeFlags) {
    auto userType = convertDTypeToUsertype(py::dtype::from_args(dType));
    PyScalarRegisterAccessor pyAcc;
    callForTypeNoVoid(userType, [&](auto&& type) {
      auto acc = _device.getScalarRegisterAccessor<std::decay_t<decltype(type)>>(
          registerPathName, elementsOffset, convertFlagsFromPython(accessModeFlags));
      pyAcc.setTE(acc);
    });
    return pyAcc;
  }

  PyOneDRegisterAccessor PyDevice::getOneDRegisterAccessor(const py::object& dType, const std::string& registerPathName,
      int numberOfElements, int elementsOffset, const py::list& accessModeFlags) {
    auto userType = convertDTypeToUsertype(py::dtype::from_args(dType));
    PyOneDRegisterAccessor pyAcc;
    callForTypeNoVoid(userType, [&](auto&& type) {
      auto acc = _device.getOneDRegisterAccessor<std::decay_t<decltype(type)>>(
          registerPathName, numberOfElements, elementsOffset, convertFlagsFromPython(accessModeFlags));
      pyAcc.setTE(acc);
    });
    return pyAcc;
  }

  PyTwoDRegisterAccessor PyDevice::getTwoDRegisterAccessor(const py::object& dType, const std::string& registerPathName,
      int numberOfElements, int elementsOffset, const py::list& accessModeFlags) {
    auto userType = convertDTypeToUsertype(py::dtype::from_args(dType));
    PyTwoDRegisterAccessor pyAcc;
    callForTypeNoVoid(userType, [&](auto&& type) {
      auto acc = _device.getTwoDRegisterAccessor<std::decay_t<decltype(type)>>(
          registerPathName, numberOfElements, elementsOffset, convertFlagsFromPython(accessModeFlags));
      pyAcc.setTE(acc);
    });
    return pyAcc;
  }

  ChimeraTK::RegisterCatalogue PyDevice::getRegisterCatalogue() {
    return _device.getRegisterCatalogue();
  }

  /*****************************************************************************************************************/

  pybind11::object PyDevice::read(const std::string& registerPath, const py::object& dtype, size_t numberOfElements,
      size_t elementsOffset, const py::list& flaglist) {
    auto reg = _device.getRegisterCatalogue().getRegister(registerPath);
    std::cout << "HIER read " << registerPath << "  " << reg.getNumberOfDimensions() << std::endl;

    if(reg.getNumberOfDimensions() == 0) {
      if(numberOfElements > 1) {
        throw ChimeraTK::logic_error("Attempting to read more than one element from scalar register");
      }
      auto acc = getScalarRegisterAccessor(dtype, registerPath, elementsOffset, flaglist);
      auto rv = acc.readAndGet();
      std::cout << rv << std::endl;
      return rv;
    }

    if(reg.getNumberOfDimensions() == 1) {
      auto acc = getOneDRegisterAccessor(dtype, registerPath, numberOfElements, elementsOffset, flaglist);
      return acc.readAndGet();
    }

    auto acc = getTwoDRegisterAccessor(dtype, registerPath, numberOfElements, elementsOffset, flaglist);
    acc.read();
    return acc.get();
  }

  /*****************************************************************************************************************/

  void PyDevice::writeArray(const std::string& registerPath, const py::array& data, const py::object& dtype,
      size_t wordOffsetInRegister, const py::list& flaglist) {
    if(data.ndim() > 2) {
      throw ChimeraTK::logic_error("Attempting to write array with more than 2 dimensions.");
    }
    std::cout << "============== writeList" << std::endl;
    size_t numberOfWords = data.shape(data.ndim() - 1);
    auto acc = getTwoDRegisterAccessor(dtype, registerPath, numberOfWords, wordOffsetInRegister, flaglist);
    acc.get() = data;
    acc.write();
  }

  /*****************************************************************************************************************/

  void PyDevice::writeList(const std::string& registerPath, const py::list& data, const py::object& dtype,
      size_t wordOffsetInRegister, const py::list& flaglist) {
    std::cout << "============== writeList" << std::endl;
    writeArray(registerPath, data.cast<py::array>(), dtype, wordOffsetInRegister, flaglist);
  }

  /*****************************************************************************************************************/

  void PyDevice::writeScalar(const std::string& registerPath, const UserTypeVariantNoVoid& data,
      const py::object& dtype, size_t wordOffsetInRegister, const py::list& flaglist) {
    std::cout << "============== writeScalar" << std::endl;
    std::visit([](auto v) { std::cout << v << std::endl; }, data);
    auto acc = getScalarRegisterAccessor(dtype, registerPath, wordOffsetInRegister, flaglist);
    acc.set(data);
    acc.write();
  }

  /*****************************************************************************************************************/

  void PyDevice::activateAsyncRead() {
    _device.activateAsyncRead();
  }

  /*****************************************************************************************************************/

  std::string PyDevice::getCatalogueMetadata(const std::string& parameterName) {
    return _device.getMetadataCatalogue().getMetadata(parameterName);
  }

  /*****************************************************************************************************************/

  void PyDevice::bind(py::module& mod) {
    py::class_<PyDevice> dev(mod, "Device");
    dev.def(py::init<const std::string&>())
        .def(py::init())
        .def("open", py::overload_cast<const std::string&>(&PyDevice::open), py::arg("aliasName"))
        .def("open", py::overload_cast<>(&PyDevice::open))
        .def("close", &PyDevice::close)
        .def("getVoidRegisterAccessor", &PyDevice::getVoidRegisterAccessor, py::arg("registerPathName"),
            py::arg("accessModeFlags") = py::list(py::list()))
        .def("getScalarRegisterAccessor", &PyDevice::getScalarRegisterAccessor, py::arg("userType"),
            py::arg("registerPathName"), py::arg("elementsOffset") = 0, py::arg("accessModeFlags") = py::list())
        .def("getOneDRegisterAccessor", &PyDevice::getOneDRegisterAccessor, py::arg("userType"),
            py::arg("registerPathName"), py::arg("numberOfElements") = 0, py::arg("elementsOffset") = 0,
            py::arg("accessModeFlags") = py::list())
        .def("getTwoDRegisterAccessor", &PyDevice::getTwoDRegisterAccessor, py::arg("userType"),
            py::arg("registerPathName"), py::arg("numberOfElements") = 0, py::arg("elementsOffset") = 0,
            py::arg("accessModeFlags") = py::list())
        .def("activateAsyncRead", &PyDevice::activateAsyncRead)
        .def("getRegisterCatalogue", &PyDevice::getRegisterCatalogue)
        .def("read", &PyDevice::read, py::arg("registerPath"), py::arg("dtype") = py::dtype::of<double>(),
            py::arg("numberOfWords") = 0, py::arg("wordOffsetInRegister") = 0, py::arg("accessModeFlags") = py::list())
        .def(
            "write",
            [](PyDevice& self, const std::string& registerPath, py::array& dataToWrite, const py::object& dtype,
                size_t wordOffsetInRegister, const py::list& flaglist) {
              self.writeArray(registerPath, dataToWrite, dtype, wordOffsetInRegister, flaglist);
            },
            py::arg("registerPath"), py::arg("dataToWrite"), py::arg("dtype") = py::dtype::of<double>(),
            py::arg("wordOffsetInRegister") = 0, py::arg("accessModeFlags") = py::list())

        .def(
            "write",
            [](PyDevice& self, const std::string& registerPath, py::list& dataToWrite, const py::object& dtype,
                size_t wordOffsetInRegister, const py::list& flaglist) {
              self.writeList(registerPath, dataToWrite, dtype, wordOffsetInRegister, flaglist);
            },
            py::arg("registerPath"), py::arg("dataToWrite"), py::arg("dtype") = py::dtype::of<double>(),
            py::arg("wordOffsetInRegister") = 0, py::arg("accessModeFlags") = py::list())
        .def(
            "write",
            [](PyDevice& self, const std::string& registerPath, UserTypeVariantNoVoid& dataToWrite,
                const py::object& dtype, size_t wordOffsetInRegister, const py::list& flaglist) {
              self.writeScalar(registerPath, dataToWrite, dtype, wordOffsetInRegister, flaglist);
            },
            py::arg("registerPath"), py::arg("dataToWrite"), py::arg("dtype") = py::dtype::of<double>(),
            py::arg("wordOffsetInRegister") = 0, py::arg("accessModeFlags") = py::list())

        .def("getCatalogueMetadata", &PyDevice::getCatalogueMetadata, py::arg("metaTag"));
  }

} // namespace ChimeraTK
