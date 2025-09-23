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
      const std::string& registerPathName, size_t elementsOffset, const py::list& accessModeFlags) {
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
      size_t numberOfElements, size_t elementsOffset, const py::list& accessModeFlags) {
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
      size_t numberOfElements, size_t elementsOffset, const py::list& accessModeFlags) {
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

    if(reg.getNumberOfDimensions() == 0) {
      if(numberOfElements > 1) {
        throw ChimeraTK::logic_error("Attempting to read more than one element from scalar register");
      }
      auto acc = getScalarRegisterAccessor(dtype, registerPath, elementsOffset, flaglist);
      return acc.readAndGet();
    }

    if(reg.getNumberOfDimensions() == 1) {
      auto acc = getOneDRegisterAccessor(dtype, registerPath, numberOfElements, elementsOffset, flaglist);
      acc.read();
      auto data = acc.get();
      if(py::isinstance<py::array>(data)) {
        // need to create a copy, because the nparray does not own its data but the accessor which does will go away
        return data.attr("copy")();
      }
      return data;
    }

    auto acc = getTwoDRegisterAccessor(dtype, registerPath, numberOfElements, elementsOffset, flaglist);
    acc.read();
    auto data = acc.get();
    if(py::isinstance<py::array>(data)) {
      // need to create a copy, because the nparray does not own its data but the accessor which does will go away
      return data.attr("copy")();
    }
    return data;
  }

  /*****************************************************************************************************************/

  void PyDevice::write2D(const std::string& registerPath,
      const UserTypeTemplateVariantNoVoid<PyTwoDRegisterAccessor::VVector>& data, size_t wordOffsetInRegister,
      const py::list& flaglist, py::object dtype) {
    size_t numberOfWords;
    std::visit(
        [&](const auto& v) {
          using UserType = typename std::remove_reference_t<decltype(v)>::value_type::value_type;
          if(dtype.is(py::none())) {
            dtype = convertUsertypeToDtype(ChimeraTK::DataType(typeid(UserType)));
          }
          numberOfWords = v.size() > 0 ? v[0].size() : 0;
        },
        data);

    if(numberOfWords == 0) {
      // there is a test checking that writing nothing is ok...
      return;
    }
    auto acc = getTwoDRegisterAccessor(dtype, registerPath, numberOfWords, wordOffsetInRegister, flaglist);
    acc.set(data);
    acc.write();
  }

  /*****************************************************************************************************************/

  void PyDevice::write1D(const std::string& registerPath,
      const UserTypeTemplateVariantNoVoid<PyOneDRegisterAccessor::Vector>& data, size_t wordOffsetInRegister,
      const py::list& flaglist, py::object dtype) {
    size_t numberOfWords;
    std::visit(
        [&](const auto& v) {
          using UserType = typename std::remove_reference_t<decltype(v)>::value_type;
          if(dtype.is(py::none())) {
            dtype = convertUsertypeToDtype(ChimeraTK::DataType(typeid(UserType)));
          }
          numberOfWords = v.size();
        },
        data);
    auto acc = getOneDRegisterAccessor(dtype, registerPath, numberOfWords, wordOffsetInRegister, flaglist);
    acc.set(data);
    acc.write();
  }

  /*****************************************************************************************************************/

  void PyDevice::writeScalar(const std::string& registerPath, const UserTypeVariantNoVoid& data,
      size_t wordOffsetInRegister, const py::list& flaglist, py::object dtype) {
    std::visit(
        [&](const auto& v) {
          using UserType = typename std::remove_reference_t<decltype(v)>;
          if(dtype.is(py::none())) {
            dtype = convertUsertypeToDtype(ChimeraTK::DataType(typeid(UserType)));
          }
        },
        data);
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
        .def("write", &PyDevice::write2D, py::arg("registerPath"), py::arg("dataToWrite"),
            py::arg("wordOffsetInRegister") = 0, py::arg("accessModeFlags") = py::list(), py::arg("dtype") = py::none())
        .def("write", &PyDevice::write1D, py::arg("registerPath"), py::arg("dataToWrite"),
            py::arg("wordOffsetInRegister") = 0, py::arg("accessModeFlags") = py::list(), py::arg("dtype") = py::none())
        .def("write", &PyDevice::writeScalar, py::arg("registerPath"), py::arg("dataToWrite"),
            py::arg("wordOffsetInRegister") = 0, py::arg("accessModeFlags") = py::list(), py::arg("dtype") = py::none())
        .def("getCatalogueMetadata", &PyDevice::getCatalogueMetadata, py::arg("metaTag"));
  }

} // namespace ChimeraTK
