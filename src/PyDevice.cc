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

#include <codecvt>
#include <variant>

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

  PyScalarRegisterAccessor PyDevice::getScalarRegisterAccessor(
      py::object& dType, const std::string& registerPathName, int elementsOffset, const py::list& accessModeFlags) {
    auto userType = convertDTypeToUsertype(py::dtype::from_args(dType));
    PyScalarRegisterAccessor pyAcc;
    callForTypeNoVoid(userType, [&](auto&& type) {
      auto acc = _device.getScalarRegisterAccessor<std::decay_t<decltype(type)>>(
          registerPathName, elementsOffset, convertFlagsFromPython(accessModeFlags));
      pyAcc.setTE(acc);
    });
    return pyAcc;
  }

  PyOneDRegisterAccessor PyDevice::getOneDRegisterAccessor(py::object& dType, const std::string& registerPathName,
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

  PyTwoDRegisterAccessor PyDevice::getTwoDRegisterAccessor(py::object& dType, const std::string& registerPathName,
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

  pybind11::array PyDevice::read(
      const std::string& registerPath, size_t numberOfElements, size_t elementsOffset, const py::list& flaglist) {
    auto reg = _device.getRegisterCatalogue().getRegister(registerPath);

    ChimeraTK::DataType usertype;
    if(!flaglist.contains(ChimeraTK::AccessMode::raw)) {
      usertype = reg.getDataDescriptor().minimumDataType();
    }
    else {
      usertype = reg.getDataDescriptor().rawDataType();
    }

    std::unique_ptr<pybind11::array> arr;

    ChimeraTK::callForTypeNoVoid(usertype, [&](auto arg) {
      using UserType = decltype(arg);
      auto acc = _device.getTwoDRegisterAccessor<UserType>(
          registerPath, numberOfElements, elementsOffset, convertFlagsFromPython(flaglist));
      acc.read();
      arr = std::make_unique<pybind11::array>(
          ChimeraTK::copyUserBufferToNpArray(acc, convertUsertypeToDtype(usertype), reg.getNumberOfDimensions()));
    });

    return *arr;
  }

  void PyDevice::write(
      const std::string& registerPath, py::array& arr, size_t elementsOffset, const py::list& flaglist) {
    auto usertype = convertDTypeToUsertype(arr.dtype());

    auto bufferTransfer = [&](auto arg) {
      auto acc = _device.getOneDRegisterAccessor<decltype(arg)>(
          registerPath, 0, elementsOffset, convertFlagsFromPython(flaglist));
      ChimeraTK::copyNpArrayToUserBuffer(acc, arr);
      acc.write();
    };

    ChimeraTK::callForTypeNoVoid(usertype, bufferTransfer);
  }

  void PyDevice::activateAsyncRead() {
    _device.activateAsyncRead();
  }

  std::string PyDevice::getCatalogueMetadata(const std::string& parameterName) {
    return _device.getMetadataCatalogue().getMetadata(parameterName);
  }

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
        .def("read", &PyDevice::read, py::arg("registerPath"), py::arg("numberOfWords") = 0,
            py::arg("wordOffsetInRegister") = 0, py::arg("accessModeFlags") = py::list())
        .def("write", &PyDevice::write, py::arg("registerPath"), py::arg("dataToWrite"),
            py::arg("wordOffsetInRegister") = 0, py::arg("accessModeFlags") = py::list())
        .def("getCatalogueMetadata", &PyDevice::getCatalogueMetadata, py::arg("metaTag"));
  }

} // namespace ChimeraTK