// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyScalarRegisterAccessor.h"

#include "PyVersionNumber.h"

#include <ChimeraTK/TransferElement.h>
#include <ChimeraTK/VariantUserTypes.h>
#include <ChimeraTK/VersionNumber.h>

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/
  /********************************************************************************************************************/

  void PyScalarRegisterAccessor::set(const UserTypeVariantNoVoid& val) {
    std::visit(
        [&](auto& acc) {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using expectedUserType = typename ACC::value_type;
          std::visit([&](auto value) { acc = ChimeraTK::userTypeToUserType<expectedUserType>(std::move(value)); }, val);
        },
        _accessor);
  }

  /********************************************************************************************************************/

  void PyScalarRegisterAccessor::setArray(const py::array& val) {
    // Note: we assume that the array has exactly one element, i.e. it is a scalar
    if(val.ndim() != 1) {
      throw std::runtime_error("PyScalarRegisterAccessor::setAndWrite: Expected a 1D array");
    }
    std::visit(
        [&](auto& acc) {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using expectedUserType = typename ACC::value_type;

          if constexpr(std::is_same_v<expectedUserType, ChimeraTK::Boolean>) {
            // Handle Boolean type specially - convert through bool
            py::array_t<bool> arr = val;
            auto directAccessArr = arr.template unchecked<1>();
            acc = static_cast<ChimeraTK::Boolean>(directAccessArr(0));
          }
          else if constexpr(std::is_same_v<expectedUserType, std::string>) {
            // Handle string type specially
            acc = val[0].cast<std::string>();
          }
          else {
            // Handle numeric types
            py::array_t<expectedUserType> arr = val;
            auto directAccessArr = arr.template unchecked<1>();
            acc = directAccessArr(0);
          }
        },
        _accessor);
  }

  /********************************************************************************************************************/

  void PyScalarRegisterAccessor::setList(const py::list& val) {
    // Convert the list to a numpy array with the correct dtype
    auto dtype = getValueType();
    auto np_array = convertPyListToNumpyArray(val, dtype);
    setArray(np_array);
  }

  /********************************************************************************************************************/

  py::object PyScalarRegisterAccessor::readAndGet() {
    read();
    return get();
  }

  /********************************************************************************************************************/

  void PyScalarRegisterAccessor::setAndWrite(const UserTypeVariantNoVoid& val, const PyVersionNumber& versionNumber) {
    auto vn = getNewVersionNumberIfNull(versionNumber);
    std::visit(
        [&](auto& acc) {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using expectedUserType = typename ACC::value_type;
          std::visit(
              [&](auto value) {
                acc.setAndWrite(ChimeraTK::userTypeToUserType<expectedUserType>(std::move(value)), vn);
              },
              val);
        },
        _accessor);
  }

  /********************************************************************************************************************/

  void PyScalarRegisterAccessor::setAndWriteArray(const py::array& val, const PyVersionNumber& versionNumber) {
    auto vn = getNewVersionNumberIfNull(versionNumber);
    // Note: we assume that the array has exactly one element, i.e. it is a scalar
    if(val.ndim() != 1) {
      throw std::runtime_error("PyScalarRegisterAccessor::setAndWrite: Expected a 1D array");
    }
    std::visit(
        [&](auto& acc) {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using expectedUserType = typename ACC::value_type;

          if constexpr(std::is_same_v<expectedUserType, ChimeraTK::Boolean>) {
            // Handle Boolean type specially - convert through bool
            py::array_t<bool> arr = val;
            auto directAccessArr = arr.template unchecked<1>();
            acc.setAndWrite(static_cast<ChimeraTK::Boolean>(directAccessArr(0)), vn);
          }
          else if constexpr(std::is_same_v<expectedUserType, std::string>) {
            // Handle string type specially
            acc.setAndWrite(val[0].cast<std::string>(), vn);
          }
          else {
            // Handle numeric types
            py::array_t<expectedUserType> arr = val;
            auto directAccessArr = arr.template unchecked<1>();
            acc.setAndWrite(directAccessArr(0), vn);
          }
        },
        _accessor);
  }

  /********************************************************************************************************************/

  py::object PyScalarRegisterAccessor::get() const {
    return std::visit(
        [&](auto& acc) {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using expectedUserType = typename ACC::value_type;
          auto data = boost::dynamic_pointer_cast<NDRegisterAccessor<expectedUserType>>(acc.getHighLevelImplElement())
                          ->accessData(0);
          if constexpr(std::is_same_v<expectedUserType, std::string>) {
            return py::cast(std::string(data));
          }
          else if constexpr(std::is_same_v<expectedUserType, ChimeraTK::Boolean>) {
            return py::dtype::of<bool>().attr("type")(bool(data));
          }
          else {
            return py::dtype::of<expectedUserType>().attr("type")(data);
          }
        },
        _accessor);
  }

  /********************************************************************************************************************/

  void PyScalarRegisterAccessor::writeIfDifferent(
      const UserTypeVariantNoVoid& val, const PyVersionNumber& versionNumber) {
    auto vn = getNewVersionNumberIfNull(versionNumber);
    std::visit(
        [&](auto& acc) {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using expectedUserType = typename ACC::value_type;
          std::visit(
              [&](auto value) {
                acc.writeIfDifferent(ChimeraTK::userTypeToUserType<expectedUserType>(std::move(value)), vn);
              },
              val);
        },
        _accessor);
  }

  /********************************************************************************************************************/

  void PyScalarRegisterAccessor::writeIfDifferentArray(const py::array& val, const PyVersionNumber& versionNumber) {
    auto vn = getNewVersionNumberIfNull(versionNumber);
    std::visit(
        [&](auto& acc) {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using expectedUserType = typename ACC::value_type;

          if constexpr(std::is_same_v<expectedUserType, ChimeraTK::Boolean>) {
            // Handle Boolean type specially - convert through bool
            py::array_t<bool> arr = val;
            auto directAccessArr = arr.template unchecked<1>();
            acc.writeIfDifferent(static_cast<ChimeraTK::Boolean>(directAccessArr(0)), vn);
          }
          else if constexpr(std::is_same_v<expectedUserType, std::string>) {
            // Handle string type specially
            acc.writeIfDifferent(val[0].cast<std::string>(), vn);
          }
          else {
            // Handle numeric types
            py::array_t<expectedUserType> arr = val;
            auto directAccessArr = arr.template unchecked<1>();
            acc.writeIfDifferent(directAccessArr(0), vn);
          }
        },
        _accessor);
  }
  /********************************************************************************************************************/

  UserTypeVariantNoVoid PyScalarRegisterAccessor::getAsCooked() {
    return std::visit([&](auto& acc) { return acc.template getAsCooked<double>(); }, _accessor);
  }

  /********************************************************************************************************************/

  void PyScalarRegisterAccessor::setAsCooked(UserTypeVariantNoVoid value) {
    std::visit([&](auto& acc) { std::visit([&](auto& val) { acc.setAsCooked(val); }, value); }, _accessor);
  }

  /********************************************************************************************************************/

  std::string PyScalarRegisterAccessor::repr(py::object& acc) const {
    std::string rep{"<PyScalarRegisterAccessor(type="};
    rep.append(py::cast<py::object>(py::cast(&acc).attr("getValueType")()).attr("__repr__")().cast<std::string>());
    rep.append(", name=");
    rep.append(py::cast(&acc).attr("getName")().cast<std::string>());
    rep.append(", data=");
    rep.append(py::cast<py::object>(py::cast(&acc).attr("__str__")()).cast<std::string>());
    rep.append(", versionNumber=");
    rep.append(py::cast<py::object>(py::cast(&acc).attr("getVersionNumber")()).attr("__repr__")().cast<std::string>());
    rep.append(", dataValidity=");
    rep.append(py::cast<py::object>(py::cast(&acc).attr("dataValidity")()).attr("__repr__")().cast<std::string>());
    rep.append(")>");
    return rep;
  }

  /********************************************************************************************************************/

  PyScalarRegisterAccessor::~PyScalarRegisterAccessor() = default;

  /********************************************************************************************************************/

  void PyScalarRegisterAccessor::bind(py::module& m) {
    py::class_<PyScalarRegisterAccessor, PyTransferElementBase> scalaracc(m, "ScalarRegisterAccessor",
        R"(Accessor class to read and write scalar registers transparently by using the accessor object like a variable.

        Conversion to and from the UserType will be handled by a data converter matching the register
        description in the map (if applicable).

        Note:
            Transfers between the device and the internal buffer need to be triggered using the read() and
            write() functions before reading from resp. after writing to the buffer.)",
        py::buffer_protocol());

    scalaracc.def(py::init<>())
        .def("read", &PyScalarRegisterAccessor::read,
            R"(Read the data from the device.

            If AccessMode.wait_for_new_data was set, this function will block until new data has arrived.
            Otherwise it still might block for a short time until the data transfer was complete.)")
        .def("readNonBlocking", &PyScalarRegisterAccessor::readNonBlocking,
            R"(Read the next value, if available in the input buffer.

            If AccessMode.wait_for_new_data was set, this function returns immediately and the return value
            indicates if a new value was available (true) or not (false).

            If AccessMode.wait_for_new_data was not set, this function is identical to read(), which will
            still return quickly. Depending on the actual transfer implementation, the backend might need to
            transfer data to obtain the current value before returning. Also this function is not guaranteed
            to be lock free. The return value will be always true in this mode.

            :return: True if new data was available, false otherwise.
            :rtype: bool)")
        .def("readLatest", &PyScalarRegisterAccessor::readLatest,
            R"(Read the latest value, discarding any other update since the last read if present.

            Otherwise this function is identical to readNonBlocking(), i.e. it will never wait for new values
            and it will return whether a new value was available if AccessMode.wait_for_new_data is set.

            :return: True if new data was available, false otherwise.
            :rtype: bool)")
        .def("write", &PyScalarRegisterAccessor::write, py::arg("versionNumber") = PyVersionNumber::getNullVersion(),
            R"(Write the data to device.

            The return value is true if old data was lost on the write transfer (e.g. due to a buffer overflow).
            In case of an unbuffered write transfer, the return value will always be false.

            :param versionNumber: Version number to use for this write operation. If not specified, a new version number is generated.
            :type versionNumber: VersionNumber)")
        .def("writeDestructively", &PyScalarRegisterAccessor::writeDestructively,
            py::arg("versionNumber") = PyVersionNumber::getNullVersion(),
            R"(Just like write(), but allows the implementation to destroy the content of the user buffer in the process.

            This is an optional optimisation, hence there is a default implementation which just calls the normal
            write(). In any case, the application must expect the user buffer of the accessor to contain
            undefined data after calling this function.

            :param versionNumber: Version number to use for this write operation. If not specified, a new version number is generated.
            :type versionNumber: VersionNumberl)")
        .def("getName", &PyScalarRegisterAccessor::getName,
            R"(Returns the name that identifies the process variable.

            :return: The register name.
            :rtype: str)")
        .def("getUnit", &PyScalarRegisterAccessor::getUnit,
            R"(Returns the engineering unit.

            If none was specified, it will default to 'n./a.'.

            :return: The engineering unit string.
            :rtype: str)")
        .def("getDescription", &PyScalarRegisterAccessor::getDescription,
            R"(Returns the description of this variable/register.

            :return: The description string.
            :rtype: str)")
        .def("getValueType", &PyScalarRegisterAccessor::getValueType,
            R"(Returns the type_info for the value type of this accessor.

            This can be used to determine the type at runtime.

            :return: Type information object.
            :rtype: type)")
        .def("getVersionNumber", &PyScalarRegisterAccessor::getVersionNumber,
            R"(Returns the version number that is associated with the last transfer.

            This refers to the last read or write operation.

            :return: The version number of the last transfer.
            :rtype: VersionNumber)")
        .def("isReadOnly", &PyScalarRegisterAccessor::isReadOnly,
            R"(Check if accessor is read only.

            This means it is readable but not writeable.

            :return: True if read only, false otherwise.
            :rtype: bool)")
        .def("isReadable", &PyScalarRegisterAccessor::isReadable,
            R"(Check if accessor is readable.

            :return: True if readable, false otherwise.
            :rtype: bool)")
        .def("isWriteable", &PyScalarRegisterAccessor::isWriteable,
            R"(Check if accessor is writeable.

            :return: True if writeable, false otherwise.
            :rtype: bool)")
        .def("getId", &PyScalarRegisterAccessor::getId,
            R"(Obtain unique ID for the actual implementation of this accessor.

            This means that e.g. two instances of ScalarRegisterAccessor created by the same call to
            Device.getScalarRegisterAccessor() will have the same ID, while two instances obtained by two
            different calls to Device.getScalarRegisterAccessor() will have a different ID even when accessing
            the very same register.

            :return: The unique transfer element ID.
            :rtype: TransferElementID)")
        .def("dataValidity", &PyScalarRegisterAccessor::dataValidity,
            R"(Return current validity of the data.

            Will always return DataValidity.ok if the backend does not support it.

            :return: The current data validity state.
            :rtype: DataValidity)")
        .def("get", &PyScalarRegisterAccessor::get,
            R"(Return the scalar value (without a previous read).

            :return: The current value in the buffer.
            :rtype: scalar)")
        .def("readAndGet", &PyScalarRegisterAccessor::readAndGet,
            R"(Convenience function to read and return the scalar value.

            :return: The value after reading from device.
            :rtype: scalar)")
        .def(
            "set", [](PyScalarRegisterAccessor& self, const UserTypeVariantNoVoid& val) { self.set(val); },
            py::arg("val"),
            R"(Set the scalar value.

            :param val: New value to set in the buffer.
            :type val: int, float, bool, or str)")
        .def(
            "set", [](PyScalarRegisterAccessor& self, const py::list& val) { self.setList(val); }, py::arg("val"),
            R"(Set the scalar value from a list.

            :param val: List containing a single value to set.
            :type val: list)")
        .def(
            "set", [](PyScalarRegisterAccessor& self, const py::array& val) { self.setArray(val); }, py::arg("val"),
            R"(Set the scalar value from a numpy array.

            :param val: Array containing a single value to set.
            :type val: ndarray)")
        .def("writeIfDifferent", &PyScalarRegisterAccessor::writeIfDifferent, py::arg("newValue"),
            py::arg("versionNumber") = PyVersionNumber::getNullVersion(),
            R"(Convenience function to set and write new value if it differs from the current value.

            The given version number is only used in case the value differs. If versionNumber is not specified,
            a new version number is generated only if the write actually takes place.

            :param newValue: New value to compare and potentially write.
            :type newValue: int, float, bool, or str
            :param versionNumber: Optional version number for the write operation.
            :type versionNumber: VersionNumber)")
        .def("setAndWrite", &PyScalarRegisterAccessor::setAndWrite, py::arg("newValue"),
            py::arg("versionNumber") = PyVersionNumber::getNullVersion(),
            R"(Convenience function to set and write new value.

            If versionNumber is not specified, a new version number is generated.

            :param newValue: New value to set and write.
            :type newValue: int, float, bool, or str
            :param versionNumber: Optional version number for the write operation.
            :type versionNumber: VersionNumber)")
        .def("getAsCooked", &PyScalarRegisterAccessor::getAsCooked,
            R"(Get the cooked values in case the accessor is a raw accessor (which does not do data conversion).

            This returns the converted data from the user buffer. It does not do any read or write transfers.

            :return: The cooked value.
            :rtype: float)")
        .def("setAsCooked", &PyScalarRegisterAccessor::setAsCooked, py::arg("value"),
            R"(Set the cooked values in case the accessor is a raw accessor (which does not do data conversion).

            This converts to raw and writes the data to the user buffer. It does not do any read or write transfers.

            :param value: The cooked value to set.
            :type value: float)")
        .def("interrupt", &PyScalarRegisterAccessor::interrupt,
            R"(Interrupt a blocking read operation.

            This will cause a blocking read to return immediately and throw an InterruptedException.)")
        .def("getAccessModeFlags", &PyScalarRegisterAccessor::getAccessModeFlags,
            R"(Return the access mode flags that were used to create this accessor.

            This can be used to determine the setting of the raw and the wait_for_new_data flags.

            :return: List of access mode flags.
            :rtype: list[AccessMode])")
        .def("isInitialised", &PyScalarRegisterAccessor::isInitialised,
            R"(Check if the accessor is initialised.

            :return: True if initialised, false otherwise.
            :rtype: bool)")
        .def("setDataValidity", &PyScalarRegisterAccessor::setDataValidity, py::arg("validity"),
            R"(Set the data validity of the accessor.

            :param validity: The data validity state to set.
            :type validity: DataValidity)")
        .def_property_readonly("dtype", &PyScalarRegisterAccessor::getValueType,
            R"(Return the dtype of the value type of this accessor.

            This can be used to determine the type at runtime.

            :return: Type information object.
            :rtype: dtype)")
        .def(
            "__getitem__",
            [](PyScalarRegisterAccessor& self, const size_t& index) {
              if(index != 0) {
                throw ChimeraTK::logic_error("PyScalarRegisterAccessor::__getitem__: Index out of range");
              }
              return self.get();
            },
            py::arg("index"),
            R"(Get the value of the scalar register accessor (same as get()).

            :param index: Must be 0 for scalar accessors.
            :type index: int
            :return: The current value.
            :rtype: scalar)")
        .def(
            "__setitem__",
            [](PyScalarRegisterAccessor& self, const size_t& index, const UserTypeVariantNoVoid& value) {
              if(index != 0) {
                throw ChimeraTK::logic_error("PyScalarRegisterAccessor::__setitem__: Index out of range");
              }
              return self.set(value);
            },
            py::arg("index"), py::arg("value"),
            R"(Set the value of the scalar register accessor (same as set()).

            :param index: Must be 0 for scalar accessors.
            :type index: int
            :param value: New value to set.
            :type value: int, float, bool, or str)")
        .def("__repr__", &PyScalarRegisterAccessor::repr);
    for(const auto& fn : PyTransferElementBase::specialFunctionsToEmulateNumeric) {
      scalaracc.def(fn.c_str(), [fn](PyScalarRegisterAccessor& acc, PyScalarRegisterAccessor& other) {
        return acc.get().attr(fn.c_str())(other.get());
      });
      scalaracc.def(fn.c_str(),
          [fn](PyScalarRegisterAccessor& acc, py::object& other) { return acc.get().attr(fn.c_str())(other); });
    }

    for(const auto& fn : PyTransferElementBase::specialAssignmentFunctionsToEmulateNumeric) {
      std::string fn_no_assign = "__" + fn.substr(3);
      scalaracc.def(fn.c_str(),
          [fn_no_assign](PyScalarRegisterAccessor& acc, PyScalarRegisterAccessor& other) -> PyScalarRegisterAccessor& {
            acc.set(acc.get().attr(fn_no_assign.c_str())(other.get()).cast<UserTypeVariantNoVoid>());
            return acc;
          });

      scalaracc.def(
          fn.c_str(), [fn_no_assign](PyScalarRegisterAccessor& acc, py::object& other) -> PyScalarRegisterAccessor& {
            acc.set(acc.get().attr(fn_no_assign.c_str())(other).cast<UserTypeVariantNoVoid>());
            return acc;
          });
    }

    for(const auto& fn : PyTransferElementBase::specialUnaryFunctionsToEmulateNumeric) {
      scalaracc.def(fn.c_str(), [fn](PyScalarRegisterAccessor& acc) { return acc.get().attr(fn.c_str())(); });
    }
  }

  /********************************************************************************************************************/

} // namespace ChimeraTK
