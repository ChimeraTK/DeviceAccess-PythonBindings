// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyDevice.h"

#include "PyOneDRegisterAccessor.h"
#include "PyVoidRegisterAccessor.h"

#include <ChimeraTK/AccessMode.h>
#include <ChimeraTK/NDRegisterAccessor.h>
#include <ChimeraTK/SupportedUserTypes.h>
#include <ChimeraTK/VoidRegisterAccessor.h>

#include <pybind11/stl.h>

#include <boost/smart_ptr/shared_ptr.hpp>

#include <variant>

namespace py = pybind11;

namespace ChimeraTK {

  // Helper to iterate over all variant alternatives
  template<typename Variant, typename Func, std::size_t Index = 0>
  void forEachTypeInVariant(Func&& func) {
    if constexpr(Index < std::variant_size_v<Variant>) {
      using T = std::variant_alternative_t<Index, Variant>;
      func.template operator()<T>();
      forEachTypeInVariant<Variant, Func, Index + 1>(std::forward<Func>(func));
    }
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
    AccessModeFlags convertedFlags;
    for(const auto& mode : accessModeFlags) {
      convertedFlags.add(mode.cast<AccessMode>());
    }
    auto acc = _device.getVoidRegisterAccessor(registerPathName, convertedFlags);
    return PyVoidRegisterAccessor{acc.getImpl()};
  }

  PyScalarRegisterAccessor PyDevice::getScalarRegisterAccessor(UserTypeVariantNoVoid& userType,
      const std::string& registerPathName, int elementsOffset, const py::list& accessModeFlags) {
    AccessModeFlags convertedFlags;
    for(const auto& mode : accessModeFlags) {
      convertedFlags.add(mode.cast<AccessMode>());
    }
    return std::visit(
        [&](auto&& type) {
          auto acc = _device.getScalarRegisterAccessor<std::decay_t<decltype(type)>>(
              registerPathName, elementsOffset, convertedFlags);
          return PyScalarRegisterAccessor{acc};
        },
        userType);
  }

  PyOneDRegisterAccessor PyDevice::getOneDRegisterAccessor(UserTypeVariantNoVoid& userType,
      const std::string& registerPathName, int numberOfElements, int elementsOffset, const py::list& accessModeFlags) {
    AccessModeFlags convertedFlags;
    for(const auto& mode : accessModeFlags) {
      convertedFlags.add(mode.cast<AccessMode>());
    }
    return std::visit(
        [&](auto&& type) {
          auto acc = _device.getOneDRegisterAccessor<std::decay_t<decltype(type)>>(
              registerPathName, numberOfElements, elementsOffset, convertedFlags);
          return PyOneDRegisterAccessor{acc};
        },
        userType);
  }

  PyTwoDRegisterAccessor PyDevice::getTwoDRegisterAccessor(UserTypeVariantNoVoid& userType,
      const std::string& registerPathName, int numberOfElements, int elementsOffset, const py::list& accessModeFlags) {
    AccessModeFlags convertedFlags;
    for(const auto& mode : accessModeFlags) {
      convertedFlags.add(mode.cast<AccessMode>());
    }
    return std::visit(
        [&](auto&& type) {
          auto acc = _device.getTwoDRegisterAccessor<std::decay_t<decltype(type)>>(
              registerPathName, numberOfElements, elementsOffset, convertedFlags);
          return PyTwoDRegisterAccessor{acc};
        },
        userType);
  }

  void PyDevice::activateAsyncRead() {
    _device.activateAsyncRead();
  }

  void PyDevice::bind(py::module& mod) {
    py::class_<PyDevice> dev(mod, "Device");
    dev.def(py::init<const std::string&>())
        .def("open", py::overload_cast<const std::string&>(&PyDevice::open))
        .def("open", py::overload_cast<>(&PyDevice::open))
        .def("close", &PyDevice::close)
        .def("getVoidAccessor", &PyDevice::getVoidRegisterAccessor)
        .def("getScalarAccessor", &PyDevice::getScalarRegisterAccessor)
        .def("getOneDAccessor", &PyDevice::getOneDRegisterAccessor)
        .def("getTwoDAccessor", &PyDevice::getTwoDRegisterAccessor)
        .def("activateAsyncRead", &PyDevice::activateAsyncRead);
  }

} // namespace ChimeraTK