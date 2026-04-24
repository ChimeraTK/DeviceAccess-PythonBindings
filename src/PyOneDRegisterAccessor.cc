// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyOneDRegisterAccessor.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <vector>

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/

  size_t PyOneDRegisterAccessor::getNElements() const {
    size_t rv;
    std::visit([&](auto& acc) { rv = acc.getNElements(); }, _accessor);
    return rv;
  }
  /********************************************************************************************************************/

  void PyOneDRegisterAccessor::set(const UserTypeTemplateVariantNoVoid<Vector>& vec) {
    std::visit(
        [&](auto& acc) {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using expectedUserType = typename ACC::value_type;
          std::visit(
              [&](const auto& vector) {
                std::vector<expectedUserType> converted(vector.size());
                std::transform(vector.begin(), vector.end(), converted.begin(),
                    [](auto v) { return userTypeToUserType<expectedUserType>(v); });
                acc = converted;
              },
              vec);
        },
        _accessor);
  }
  /********************************************************************************************************************/

  py::object PyOneDRegisterAccessor::readAndGet() {
    read();
    return get();
  }

  /********************************************************************************************************************/

  void PyOneDRegisterAccessor::setAndWrite(
      const UserTypeTemplateVariantNoVoid<Vector>& vec, const PyVersionNumber& versionNumber) {
    set(vec);
    auto vn = getNewVersionNumberIfNull(versionNumber);
    write(vn);
  }
  /********************************************************************************************************************/

  py::object PyOneDRegisterAccessor::get() const {
    return std::visit(
        [&](auto& acc) -> py::object {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using userType = typename ACC::value_type;
          auto ndacc = boost::dynamic_pointer_cast<NDRegisterAccessor<userType>>(acc.getHighLevelImplElement());
          if constexpr(std::is_same<userType, std::string>::value) {
            // String arrays are not really supported by numpy, so we return a list instead
            return py::cast(ndacc->accessChannel(0));
          }
          else if constexpr(std::is_same<userType, ChimeraTK::Boolean>::value) {
            auto ary = py::array(py::dtype::of<bool>(), {acc.getNElements()}, {sizeof(userType)},
                ndacc->accessChannel(0).data(), py::cast(this));
            assert(!ary.owndata()); // numpy must not own our buffers
            return ary;
          }
          else {
            auto ary = py::array(py::dtype::of<userType>(), {acc.getNElements()}, {sizeof(userType)},
                ndacc->accessChannel(0).data(), py::cast(this));
            assert(!ary.owndata()); // numpy must not own our buffers
            return ary;
          }
        },
        _accessor);
  }

  /********************************************************************************************************************/

  py::object PyOneDRegisterAccessor::getitem(size_t index) const {
    if(index >= getNElements()) {
      // throwing this exception is actually required, because Python sometimes seems to iterate until the exception
      // is thrown (e.g. when the accessor is used inside zip()).
      throw std::out_of_range("Index out of range");
    }
    py::object rv;
    std::visit([&](auto& acc) { rv = py::cast(acc[index]); }, _accessor);
    return rv;
  }

  /********************************************************************************************************************/

  void PyOneDRegisterAccessor::setitem(size_t index, const UserTypeVariantNoVoid& val) {
    std::visit(
        [&](auto& acc) {
          std::visit(
              [&](auto& v) {
                acc[index] = userTypeToUserType<typename std::remove_reference<decltype(acc)>::type::value_type>(v);
              },
              val);
        },
        _accessor);
  }
  /********************************************************************************************************************/

  UserTypeVariantNoVoid PyOneDRegisterAccessor::getAsCooked(uint element) {
    return std::visit([&](auto& acc) { return acc.template getAsCooked<double>(element); }, _accessor);
  }

  /********************************************************************************************************************/

  void PyOneDRegisterAccessor::setAsCooked(uint element, UserTypeVariantNoVoid value) {
    std::visit([&](auto& acc) { std::visit([&](auto& val) { acc.setAsCooked(element, val); }, value); }, _accessor);
  }

  /********************************************************************************************************************/

  std::string PyOneDRegisterAccessor::repr(py::object& acc) const {
    std::string rep{"<PyOneDRegisterAccessor(type="};
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

  py::buffer_info PyOneDRegisterAccessor::getBufferInfo() {
    py::buffer_info info;
    std::visit(
        [&](auto& acc) {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using userType = typename ACC::value_type;
          auto ndacc = boost::dynamic_pointer_cast<NDRegisterAccessor<userType>>(acc.getHighLevelImplElement());
          if constexpr(std::is_same<userType, ChimeraTK::Boolean>::value) {
            info.format = py::format_descriptor<bool>::format();
          }
          else if constexpr(std::is_same<userType, std::string>::value) {
            // cannot implement
            return;
          }
          else {
            info.format = py::format_descriptor<userType>::format();
          }
          info.ptr = ndacc->accessChannel(0).data();
          info.itemsize = sizeof(userType);
          info.ndim = 1;
          info.shape = {acc.getNElements()};
          info.strides = {sizeof(userType)};
        },
        _accessor);
    return info;
  }

  /********************************************************************************************************************/

  PyOneDRegisterAccessor::~PyOneDRegisterAccessor() = default;

  /********************************************************************************************************************/

  void PyOneDRegisterAccessor::bind(py::module& m) {
    py::class_<PyOneDRegisterAccessor, PyTransferElementBase> arrayacc(m, "OneDRegisterAccessor",
        R"(Accessor class to read and write registers transparently by using the accessor object like a numpy array.

        Conversion to and from the UserType will be handled by a data converter matching the register
        description in the map (if applicable).

        Note:
            Transfers between the device and the internal buffer need to be triggered using the read() and
            write() functions before reading from resp. after writing to the buffer.)",
        py::buffer_protocol());

    arrayacc.def(py::init<>())
        .def_buffer(&PyOneDRegisterAccessor::getBufferInfo)
        .def("read", &PyOneDRegisterAccessor::read,
            R"(Read the data from the device.

            If AccessMode.wait_for_new_data was set, this function will block until new data has arrived.
            Otherwise it still might block for a short time until the data transfer was complete.

          See Also:
              readNonBlocking: Read without blocking if no data is available.
              readLatest: Read latest value while discarding intermediate updates.
              readAndGet: Convenience method combining read() and get().

          Returns:
            None: This function does not return a value.)")
        .def("readNonBlocking", &PyOneDRegisterAccessor::readNonBlocking,
            R"(Read the next value, if available in the input buffer.

            If AccessMode.wait_for_new_data was set, this function returns immediately and the return value
            indicates if a new value was available (true) or not (false).

            If AccessMode.wait_for_new_data was not set, this function is identical to read(), which will
            still return quickly. Depending on the actual transfer implementation, the backend might need to
            transfer data to obtain the current value before returning. Also this function is not guaranteed
            to be lock free. The return value will be always true in this mode.

          Returns:
            bool: True if new data was available, false otherwise.)")
        .def("readLatest", &PyOneDRegisterAccessor::readLatest,
            R"(Read the latest value, discarding any other update since the last read if present.

            Otherwise this function is identical to readNonBlocking(), i.e. it will never wait for new values
            and it will return whether a new value was available if AccessMode.wait_for_new_data is set.

          Returns:
            bool: True if new data was available, false otherwise.)")
        .def("write", &PyOneDRegisterAccessor::write, py::arg("versionNumber") = PyVersionNumber::getNullVersion(),
            R"(Write the data to device.

            The return value is true if old data was lost on the write transfer (e.g. due to a buffer overflow).
            In case of an unbuffered write transfer, the return value will always be false.

          Args:
            versionNumber (VersionNumber): Version number to use for this write operation. If not specified,
              a new version number is generated.

          Returns:
            bool: True if data was lost, false otherwise.

          See Also:
              setAndWrite: Convenience method combining set() and write().
              writeDestructively: Optimized write that may destroy buffer.)")
        .def("writeDestructively", &PyOneDRegisterAccessor::writeDestructively,
            py::arg("versionNumber") = PyVersionNumber::getNullVersion(),
            R"(Just like write(), but allows the implementation to destroy the content of the user buffer in the process.

            This is an optional optimisation, hence there is a default implementation which just calls the normal
            write(). In any case, the application must expect the user buffer of the accessor to contain
            undefined data after calling this function.

          Args:
            versionNumber (VersionNumber): Version number to use for this write operation. If not specified,
              a new version number is generated.

          Returns:
            bool: True if data was lost, false otherwise.)")
        .def("interrupt", &PyOneDRegisterAccessor::interrupt,
            R"(Interrupt a blocking read operation.

            This will cause a blocking read to return immediately and throw an InterruptedException.

          Returns:
            None: This function does not return a value.)")
        .def("getName", &PyOneDRegisterAccessor::getName,
            R"(Return the name that identifies the process variable.

          Returns:
            str: The register name.)")
        .def("getUnit", &PyOneDRegisterAccessor::getUnit,
            R"(Return the engineering unit.

            If none was specified, it will default to 'n./a.'.

          Returns:
            str: The engineering unit string.)")
        .def("getDescription", &PyOneDRegisterAccessor::getDescription,
            R"(Return the description of this variable/register.

          Returns:
            str: The description string.)")
        .def("getValueType", &PyOneDRegisterAccessor::getValueType,
            R"(Return the numpy dtype for the value type of this accessor.

            This can be used to determine the type at runtime.

          Returns:
            numpy.dtype: Type information object.)")
        .def("getVersionNumber", &PyOneDRegisterAccessor::getVersionNumber,
            R"(Return the version number that is associated with the last transfer.

            This refers to the last read or write operation.

          Returns:
            VersionNumber: The version number of the last transfer.)")
        .def("isReadOnly", &PyOneDRegisterAccessor::isReadOnly,
            R"(Check if accessor is read only.

            This means it is readable but not writeable.

          Returns:
            bool: True if read only, false otherwise.)")
        .def("isReadable", &PyOneDRegisterAccessor::isReadable,
            R"(Check if accessor is readable.

          Returns:
            bool: True if readable, false otherwise.)")
        .def("isWriteable", &PyOneDRegisterAccessor::isWriteable,
            R"(Check if accessor is writeable.

          Returns:
            bool: True if writeable, false otherwise.)")
        .def("getId", &PyOneDRegisterAccessor::getId,
            R"(Obtain unique ID for the actual implementation of this accessor.

            This means that e.g. two instances of OneDRegisterAccessor created by the same call to
            Device.getOneDRegisterAccessor() will have the same ID, while two instances obtained by two
            different calls to Device.getOneDRegisterAccessor() will have a different ID even when accessing
            the very same register.

          Returns:
            TransferElementID: The unique accessor ID.)")
        .def("dataValidity", &PyOneDRegisterAccessor::dataValidity,
            R"(Return current validity of the data.

            Will always return DataValidity.ok if the backend does not support it.

          Returns:
            DataValidity: The current data validity state.)")
        .def("getNElements", &PyOneDRegisterAccessor::getNElements,
            R"(Return number of elements/samples in the register.

          Returns:
            int: Number of elements in the register.)")
        .def("get", &PyOneDRegisterAccessor::get,
            R"(Return the register data as an array (without a previous read).

          Returns:
            ndarray: Array containing the register data.)")
        .def("set", &PyOneDRegisterAccessor::set, py::arg("newValue"),
            R"(Set the values of the array.

          Args:
            newValue (list | ndarray): New values to set in the buffer.

          Returns:
            None: This function does not return a value.)")
        .def("setAndWrite", &PyOneDRegisterAccessor::setAndWrite, py::arg("newValue"),
            py::arg("versionNumber") = PyVersionNumber::getNullVersion(),
            R"(Convenience function to set and write new value.

            If versionNumber is not specified, a new version number is generated.

          Args:
            newValue (list | ndarray): New values to set and write.
            versionNumber (VersionNumber): Optional version number for the write operation.

          Returns:
            None: This function does not return a value.)")
        .def("getAsCooked", &PyOneDRegisterAccessor::getAsCooked, py::arg("element"),
            R"(Get the cooked values in case the accessor is a raw accessor (which does not do data conversion).

            This returns the converted data from the user buffer. It does not do any read or write transfer.

          Args:
            element (int): Element index to read.

          Returns:
            int: The cooked value at the specified element.)")
        .def("setAsCooked", &PyOneDRegisterAccessor::setAsCooked, py::arg("element"), py::arg("value"),
            R"(Set the cooked values in case the accessor is a raw accessor (which does not do data conversion).

            This converts to raw and writes the data to the user buffer. It does not do any read or write transfer.

          Args:
            element (int): Element index to write.
            value (float): The cooked value to set.

          Returns:
            None: This function does not return a value.)")
        .def("isInitialised", &PyOneDRegisterAccessor::isInitialised,
            R"(Check if the accessor is initialised.

          Returns:
            bool: True if initialised, false otherwise.)")
        .def("setDataValidity", &PyOneDRegisterAccessor::setDataValidity, py::arg("validity"),
            R"(Set the data validity of the accessor.

            Args:
              validity (DataValidity): The data validity state to set.

            Returns:
              None: This function does not return a value.)")
        .def("getAccessModeFlags", &PyOneDRegisterAccessor::getAccessModeFlags,
            R"(Return the access mode flags that were used to create this accessor.

            This can be used to determine the setting of the raw and the wait_for_new_data flags.

            Returns:
              list[AccessMode]: List of access mode flags.)")
        .def("readAndGet", &PyOneDRegisterAccessor::readAndGet,
            R"(Convenience function to read and return the register data.

            Returns:
              ndarray: Array containing the register data after reading.)")
        .def(
            "__getitem__", [](PyOneDRegisterAccessor& acc, size_t index) { return acc.getitem(index); },
            py::arg("index"),
            R"(Get an element from the array by index.

            Args:
              index (int): The element index.

            Returns:
              scalar: The value at the specified index.)")
        .def(
            "__getitem__",
            [](PyOneDRegisterAccessor& acc, const py::object& slice) { return acc.get().attr("__getitem__")(slice); },
            py::arg("slice"),
            R"(Get an element from the array by index.

            Args:
              slice (slice): A slice object.

            Returns:
              np.ndarray: The value(s) at the specified slice.)")
        .def(
            "__setitem__",
            [](PyOneDRegisterAccessor& acc, size_t index, const UserTypeVariantNoVoid& value) {
              acc.setitem(index, value);
            },
            py::arg("index"), py::arg("value"),
            R"(Set an element in the array by index.

            Args:
              index (int): The element index.
              value (user type): The value to set at the specified index.

            Returns:
              None: This function does not return a value.)")
        .def(
            "__setitem__",
            [](PyOneDRegisterAccessor& acc, const py::object& slice, const UserTypeVariantNoVoid& value) {
              acc.get().attr("__setitem__")(slice, py::cast(value));
            },
            py::arg("slice"), py::arg("value"),
            R"(Set an element in the array by slice.

            Args:
              slice (slice): The element slice.
              value (user type): The value to set at the specified slice.

            Returns:
              None: This function does not return a value.)")
        .def(
            "__setitem__",
            [](PyOneDRegisterAccessor& acc, const py::object& slice, const py::object& array) {
              acc.get().attr("__setitem__")(slice, array);
            },
            py::arg("slice"), py::arg("array"),
            R"(Set an element in the array by slice.

            Args:
              slice (slice): The element slice.
              array (list or ndarray): The value to set at the specified slice.

            Returns:
              None: This function does not return a value.)")
        .def("__getattr__", &PyOneDRegisterAccessor::getattr, py::arg("name"),
            R"(Forward unknown attribute access to the underlying array-like object.

            Args:
              name (str): Name of the attribute.

            Returns:
              object: Attribute value or callable attribute proxy.)");

    for(const auto& fn : PyTransferElementBase::specialFunctionsToEmulateNumeric) {
      arrayacc.def(fn.c_str(), [fn](PyOneDRegisterAccessor& acc, PyOneDRegisterAccessor& other) {
        return acc.get().attr(fn.c_str())(other.get());
      });
      arrayacc.def(fn.c_str(),
          [fn](PyOneDRegisterAccessor& acc, py::object& other) { return acc.get().attr(fn.c_str())(other); });
    }

    for(const auto& fn : PyTransferElementBase::specialAssignmentFunctionsToEmulateNumeric) {
      arrayacc.def(
          fn.c_str(), [fn](PyOneDRegisterAccessor& acc, PyOneDRegisterAccessor& other) -> PyOneDRegisterAccessor& {
            acc.get().attr(fn.c_str())(other.get());
            return acc;
          });

      arrayacc.def(fn.c_str(), [fn](PyOneDRegisterAccessor& acc, py::object& other) -> PyOneDRegisterAccessor& {
        acc.get().attr(fn.c_str())(other);
        return acc;
      });
    }

    for(const auto& fn : PyTransferElementBase::specialUnaryFunctionsToEmulateNumeric) {
      arrayacc.def(fn.c_str(), [fn](PyOneDRegisterAccessor& acc) { return acc.get().attr(fn.c_str())(); });
    }
  }

  /********************************************************************************************************************/

} // namespace ChimeraTK
