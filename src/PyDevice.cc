// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyDevice.h"

#include "HelperFunctions.h"
#include "PyOneDRegisterAccessor.h"
#include "PyVoidRegisterAccessor.h"

#include <ChimeraTK/AccessMode.h>
#include <ChimeraTK/NDRegisterAccessor.h>
#include <ChimeraTK/SupportedUserTypes.h>
#include <ChimeraTK/VoidRegisterAccessor.h>

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

  // Helper function to convert ChimeraTK::DataType to py::dtype

  py::dtype convertUsertypeToDtype(ChimeraTK::DataType usertype) {
    std::unique_ptr<py::dtype> rv;
    ChimeraTK::callForTypeNoVoid(usertype, [&](auto arg) {
      using UserType = decltype(arg);
      if constexpr(std::is_same<UserType, ChimeraTK::Boolean>::value) {
        rv = std::make_unique<py::dtype>(py::dtype::of<bool>());
      }
      if constexpr(std::is_same<UserType, std::string>::value) {
        rv = std::make_unique<py::dtype>(py::dtype::of<char*>());
      }
      else {
        rv = std::make_unique<py::dtype>(py::dtype::of<UserType>());
      }
    });
    return *rv;
  }

  // Helper function to copy data from user buffer to numpy dtype
  ChimeraTK::DataType convertDytpeToUsertype(py::dtype dtype) {
    if(dtype.is(py::dtype::of<int8_t>())) {
      return ChimeraTK::DataType::int8;
    }
    if(dtype.is(py::dtype::of<int16_t>())) {
      return ChimeraTK::DataType::int16;
    }
    if(dtype.is(py::dtype::of<int32_t>())) {
      return ChimeraTK::DataType::int32;
    }
    if(dtype.is(py::dtype::of<int64_t>())) {
      return ChimeraTK::DataType::int64;
    }
    if(dtype.is(py::dtype::of<uint8_t>())) {
      return ChimeraTK::DataType::uint8;
    }
    if(dtype.is(py::dtype::of<uint16_t>())) {
      return ChimeraTK::DataType::uint16;
    }
    if(dtype.is(py::dtype::of<uint32_t>())) {
      return ChimeraTK::DataType::uint32;
    }
    if(dtype.is(py::dtype::of<uint64_t>())) {
      return ChimeraTK::DataType::uint64;
    }
    if(dtype.is(py::dtype::of<float>())) {
      return ChimeraTK::DataType::float32;
    }
    if(dtype.is(py::dtype::of<double>())) {
      return ChimeraTK::DataType::float64;
    }
    if(dtype.is(py::dtype::of<bool>())) {
      return ChimeraTK::DataType::Boolean;
    }
    throw std::invalid_argument("Unsupported numpy dtype");
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

  PyScalarRegisterAccessor PyDevice::getScalarRegisterAccessor(UserTypeVariantNoVoid& userType,
      const std::string& registerPathName, int elementsOffset, const py::list& accessModeFlags) {
    return std::visit(
        [&](auto&& type) {
          auto acc = _device.getScalarRegisterAccessor<std::decay_t<decltype(type)>>(
              registerPathName, elementsOffset, convertFlagsFromPython(accessModeFlags));
          return PyScalarRegisterAccessor{acc};
        },
        userType);
  }

  PyOneDRegisterAccessor PyDevice::getOneDRegisterAccessor(UserTypeVariantNoVoid& userType,
      const std::string& registerPathName, int numberOfElements, int elementsOffset, const py::list& accessModeFlags) {
    return std::visit(
        [&](auto&& type) {
          auto acc = _device.getOneDRegisterAccessor<std::decay_t<decltype(type)>>(
              registerPathName, numberOfElements, elementsOffset, convertFlagsFromPython(accessModeFlags));
          return PyOneDRegisterAccessor{acc};
        },
        userType);
  }

  PyTwoDRegisterAccessor PyDevice::getTwoDRegisterAccessor(UserTypeVariantNoVoid& userType,
      const std::string& registerPathName, int numberOfElements, int elementsOffset, const py::list& accessModeFlags) {
    return std::visit(
        [&](auto&& type) {
          auto acc = _device.getTwoDRegisterAccessor<std::decay_t<decltype(type)>>(
              registerPathName, numberOfElements, elementsOffset, convertFlagsFromPython(accessModeFlags));
          return PyTwoDRegisterAccessor{acc};
        },
        userType);
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
      arr = std::make_unique<pybind11::array>(DeviceAccessPython::copyUserBufferToNpArray(
          acc, convertUsertypeToDtype(usertype), reg.getNumberOfDimensions()));
    });

    return *arr;
  }

  void PyDevice::write(py::array& arr, const std::string& registerPath, size_t numberOfElements, size_t elementsOffset,
      const py::list& flaglist) {
    auto usertype = convertDytpeToUsertype(arr.dtype());

    auto bufferTransfer = [&](auto arg) {
      auto acc = _device.getTwoDRegisterAccessor<decltype(arg)>(
          registerPath, numberOfElements, elementsOffset, convertFlagsFromPython(flaglist));
      DeviceAccessPython::copyNpArrayToUserBuffer(acc, arr);
      acc.write();
    };

    ChimeraTK::callForTypeNoVoid(usertype, bufferTransfer);
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