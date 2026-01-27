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
    py::class_<PyDevice> dev(mod, "Device",
        R"(Class to access a ChimeraTK device.

      The device can be opened and closed, and provides methods to obtain register accessors. Additionally,
      convenience methods to read and write registers directly are provided.
      The class also offers methods to check the device state, obtain the register catalogue,  Metadata and to set exception conditions.)");

    dev.def(py::init<const std::string&>(), py::arg("aliasName"),
           R"(Initialize device and associate a backend.

          Note:
              The device is not opened after initialization.

          Args:
              aliasName: The ChimeraTK device descriptor for the device.)")
        .def(py::init(),
            R"(Create device instance without associating a backend yet.

          A backend has to be explicitly associated using open method which
          has the alias or CDD as argument.)")
        .def("open", py::overload_cast<const std::string&>(&PyDevice::open), py::arg("aliasName"),
            R"(Open a device by the given alias name from the DMAP file.

          Args:
              aliasName (str): The device alias name from the DMAP file.)")
        .def("open", py::overload_cast<>(&PyDevice::open),
            R"((Re-)Open the device.

          Can only be called when the device was constructed with a given aliasName.)")
        .def("close", &PyDevice::close,
            R"(Close the device.

          The connection with the alias name is kept so the device can be re-opened
          using the open() function without argument.)")
        .def("getVoidRegisterAccessor", &PyDevice::getVoidRegisterAccessor, py::arg("registerPathName"),
            py::arg("accessModeFlags") = py::list(),
            R"(Get a VoidRegisterAccessor object for the given register.

          Args:
              registerPathName (str): Full path name of the register.
              accessModeFlags (list[:class:`AccessMode`]): Optional flags to control register access details.

          Returns:
            :class:`VoidRegisterAccessor`: VoidRegisterAccessor for the specified register.

          See Also:
            :meth:`getScalarRegisterAccessor`: For scalar value registers.
            :meth:`getOneDRegisterAccessor`: For 1D array registers.
            :meth:`getTwoDRegisterAccessor`: For 2D array registers.
            :meth:`getRegisterCatalogue`: Browse all available registers.)")
        .def("getScalarRegisterAccessor", &PyDevice::getScalarRegisterAccessor, py::arg("userType"),
            py::arg("registerPathName"), py::arg("elementsOffset") = 0, py::arg("accessModeFlags") = py::list(),
            R"(Get a ScalarRegisterAccessor object for the given register.

          The ScalarRegisterAccessor allows to read and write registers transparently
          by using the accessor object like a variable of the specified data type.

          Args:
              userType (numpy.dtype): The data type for register access (numpy dtype).
              registerPathName (str): Full path name of the register.
              elementsOffset (int): Word offset in register to access other than the first word.
              accessModeFlags (list[:class:`AccessMode`]): Optional flags to control register access details.

          Returns:
            :class:`ScalarRegisterAccessor`: ScalarRegisterAccessor for the specified register.

          See Also:
            :meth:`getOneDRegisterAccessor`: For 1D array registers.
            :meth:`getTwoDRegisterAccessor`: For 2D array registers.
            :meth:`getVoidRegisterAccessor`: For trigger-only registers.
            :meth:`read`: Convenience function for one-time reads.
            :meth:`write`: Convenience function for one-time writes.)")
        .def("getOneDRegisterAccessor", &PyDevice::getOneDRegisterAccessor, py::arg("userType"),
            py::arg("registerPathName"), py::arg("numberOfElements") = 0, py::arg("elementsOffset") = 0,
            py::arg("accessModeFlags") = py::list(),
            R"(Get a OneDRegisterAccessor object for the given register.

          The OneDRegisterAccessor allows to read and write registers transparently
          by using the accessor object like a vector of the specified data type.

          Args:
              userType (numpy.dtype): The data type for register access (numpy dtype).
              registerPathName (str): Full path name of the register.
              numberOfElements (int): Number of elements to access (0 for entire register).
              elementsOffset (int): Word offset in register to skip initial elements.
              accessModeFlags (list[:class:`AccessMode`]): Optional flags to control register access details.

          Returns:
            :class:`OneDRegisterAccessor`: OneDRegisterAccessor for the specified register.

          See Also:
            :meth:`getScalarRegisterAccessor`: For single-value registers.
            :meth:`getTwoDRegisterAccessor`: For 2D array registers.
            :meth:`getVoidRegisterAccessor`: For trigger-only registers.
            :meth:`read`: Convenience function for one-time reads.
            :meth:`write`: Convenience function for one-time writes.)")
        .def("getTwoDRegisterAccessor", &PyDevice::getTwoDRegisterAccessor, py::arg("userType"),
            py::arg("registerPathName"), py::arg("numberOfElements") = 0, py::arg("elementsOffset") = 0,
            py::arg("accessModeFlags") = py::list(),
            R"(Get a TwoDRegisterAccessor object for the given register.

          This allows to read and write transparently 2-dimensional registers.

          Args:
              userType (numpy.dtype): The data type for register access (numpy dtype).
              registerPathName (str): Full path name of the register.
              numberOfElements (int): Number of elements per channel (0 for all).
              elementsOffset (int): First element index for each channel to read.
              accessModeFlags (list[:class:`AccessMode`]): Optional flags to control register access details.

          Returns:
            :class:`TwoDRegisterAccessor`: TwoDRegisterAccessor for the specified register.

          See Also:
            :meth:`getOneDRegisterAccessor`: For 1D array registers.
            :meth:`getScalarRegisterAccessor`: For single-value registers.
            :meth:`getVoidRegisterAccessor`: For trigger-only registers.
            :meth:`read`: Convenience function for one-time reads.
            :meth:`write`: Convenience function for one-time writes.)")
        .def("activateAsyncRead", &PyDevice::activateAsyncRead,
            R"(Activate asynchronous read for all transfer elements with wait_for_new_data flag.

          If called while the device is not opened or has an error, this call has no effect.
          When this function returns, it is not guaranteed that all initial values have been
          received already.)")
        .def("getRegisterCatalogue", &PyDevice::getRegisterCatalogue,
            R"(Return the register catalogue with detailed information on all registers.

          Returns:
              :class:`RegisterCatalogue`: RegisterCatalogue containing all register information.)")
        .def("read", &PyDevice::read, py::arg("registerPath"), py::arg("dtype") = py::dtype::of<double>(),
            py::arg("numberOfWords") = 0, py::arg("wordOffsetInRegister") = 0, py::arg("accessModeFlags") = py::list(),
            R"(Convenience function to read a register without obtaining an accessor.

          Warning:
              This function is inefficient as it creates and discards a register
              accessor in each call. For better performance, use register accessors instead.

          Args:
              registerPath (str): Full path name of the register.
              dtype (numpy.dtype): Data type for the read operation (default: float64).
              numberOfWords (int): Number of elements to read (0 for scalar or entire register).
              wordOffsetInRegister (int): Word offset in register to skip initial elements.
              accessModeFlags (list[:class:`AccessMode`]): Optional flags to control register access details.

          Returns:
            scalar | ndarray: Register value (scalar, 1D array, or 2D array depending on register type).

          See Also:
            :meth:`getScalarRegisterAccessor`: For efficient repeated scalar access.
            :meth:`getOneDRegisterAccessor`: For efficient repeated 1D array access.
            :meth:`getTwoDRegisterAccessor`: For efficient repeated 2D array access.
            :meth:`write`: Convenience function for one-time writes.)")
        .def("write", &PyDevice::write2D, py::arg("registerPath"), py::arg("dataToWrite"),
            py::arg("wordOffsetInRegister") = 0, py::arg("accessModeFlags") = py::list(), py::arg("dtype") = py::none())
        .def("write", &PyDevice::write1D, py::arg("registerPath"), py::arg("dataToWrite"),
            py::arg("wordOffsetInRegister") = 0, py::arg("accessModeFlags") = py::list(), py::arg("dtype") = py::none())
        .def("write", &PyDevice::writeScalar, py::arg("registerPath"), py::arg("dataToWrite"),
            py::arg("wordOffsetInRegister") = 0, py::arg("accessModeFlags") = py::list(), py::arg("dtype") = py::none(),
            R"(Convenience function to write a register without obtaining an accessor.

          This method is overloaded to handle different data types. The appropriate overload is selected based
          on the type of dataToWrite.

          Warning:
              This function is inefficient as it creates and discards a register accessor in each call.
              For better performance, use register accessors instead:
              :meth:`getScalarRegisterAccessor`, :meth:`getOneDRegisterAccessor`, :meth:`getTwoDRegisterAccessor`.

          Args:
              registerPath (str): Full path name of the register.
              dataToWrite (int | float | bool | str | ndarray): Data to write. Type determines operation:

                  - Scalar (int, float, bool, str): Write scalar value to single-element register.
                  - 1D array (ndarray): Write 1D array data to 1D register.
                  - 2D array (ndarray): Write 2D array data to 2D register.

              wordOffsetInRegister (int): Word offset in register to skip initial elements (default: 0).
              accessModeFlags (list[:class:`AccessMode`]): Optional flags to control register access (default: []).
              dtype (numpy.dtype | None): Optional data type override. If None, type is inferred from data (default: None).

          Examples:
              >>> import ChimeraTK.DeviceAccess as da
              >>> import numpy as np
              >>> da.setDMapFilePath('testCrate.dmap')
              >>> device = da.Device('TEST_CARD')
              >>> device.open()
              >>> # Write scalar value
              >>> device.write('MYSCALAR', 42.5)
              >>> # Write 1D array
              >>> device.write('MYARRAY', np.array([1.0, 2.0, 3.0]))
              >>> # Write 2D array
              >>> device.write('MY2DARRAY', np.array([[1, 2], [3, 4]]))
          See Also:
              :meth:`getScalarRegisterAccessor`: For efficient repeated scalar access.
              :meth:`getOneDRegisterAccessor`: For efficient repeated 1D array access.
              :meth:`getTwoDRegisterAccessor`: For efficient repeated 2D array access.
              :meth:`read`: Convenience function for one-time reads.)")
        .def(
            "isOpened", [](PyDevice& self) { return self._device.isOpened(); },
            R"(Check if the device is currently opened.

          Returns:
              bool: True if device is opened, False otherwise.)")
        .def(
            "setException", [](PyDevice& self, const std::string& msg) { return self._device.setException(msg); },
            py::arg("message"),
            R"(Set the device into an exception state.

          All asynchronous reads will be deactivated and all operations will see
          exceptions until open() has successfully been called again.

          Args:
              message (str): Exception message describing the error condition.)")
        .def(
            "isFunctional", [](PyDevice& self) { return self._device.isFunctional(); },
            R"(Check whether the device is working as intended.

          Usually this means it is opened and does not have any errors.

          Returns:
              bool: True if device is functional, False otherwise.)")
        .def("getCatalogueMetadata", &PyDevice::getCatalogueMetadata, py::arg("metaTag"),
            R"(Get metadata from the device catalogue.

          Args:
              metaTag (str): The metadata parameter name to retrieve.

          Returns:
              str: The metadata value.)")
        .def("__enter__",
            [](PyDevice& self) {
              self.open();
              return &self;
            })
        .def("__exit__",
            [](PyDevice& self, [[maybe_unused]] py::object exc_type, [[maybe_unused]] py::object exc_val,
                [[maybe_unused]] py::object exc_traceback) {
              self.close();
              return false;
            });
  }

} // namespace ChimeraTK
