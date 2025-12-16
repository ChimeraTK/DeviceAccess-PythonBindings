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
            Otherwise it still might block for a short time until the data transfer was complete.)")
        .def("readNonBlocking", &PyOneDRegisterAccessor::readNonBlocking,
            R"(Read the next value, if available in the input buffer.

            If AccessMode.wait_for_new_data was set, this function returns immediately and the return value
            indicates if a new value was available (true) or not (false).

            If AccessMode.wait_for_new_data was not set, this function is identical to read(), which will
            still return quickly. Depending on the actual transfer implementation, the backend might need to
            transfer data to obtain the current value before returning. Also this function is not guaranteed
            to be lock free. The return value will be always true in this mode.

            :return: True if new data was available, false otherwise.
            :rtype: bool)")
        .def("readLatest", &PyOneDRegisterAccessor::readLatest,
            R"(Read the latest value, discarding any other update since the last read if present.

            Otherwise this function is identical to readNonBlocking(), i.e. it will never wait for new values
            and it will return whether a new value was available if AccessMode.wait_for_new_data is set.

            :return: True if new data was available, false otherwise.
            :rtype: bool)")
        .def("write", &PyOneDRegisterAccessor::write, py::arg("versionNumber") = PyVersionNumber::getNullVersion(),
            R"(Write the data to device.

            The return value is true if old data was lost on the write transfer (e.g. due to a buffer overflow).
            In case of an unbuffered write transfer, the return value will always be false.

            :param versionNumber: Version number to use for this write operation. If not specified, a new version number is generated.
            :type versionNumber: VersionNumber
            :return: True if data was lost, false otherwise.
            :rtype: bool)")
        .def("writeDestructively", &PyOneDRegisterAccessor::writeDestructively,
            py::arg("versionNumber") = PyVersionNumber::getNullVersion(),
            R"(Just like write(), but allows the implementation to destroy the content of the user buffer in the process.

            This is an optional optimisation, hence there is a default implementation which just calls the normal
            write(). In any case, the application must expect the user buffer of the accessor to contain
            undefined data after calling this function.

            :param versionNumber: Version number to use for this write operation. If not specified, a new version number is generated.
            :type versionNumber: VersionNumber
            :return: True if data was lost, false otherwise.
            :rtype: bool)")
        .def("interrupt", &PyOneDRegisterAccessor::interrupt,
            R"(Interrupt a blocking read operation.

            This will cause a blocking read to return immediately and throw an InterruptedException.)")
        .def("getName", &PyOneDRegisterAccessor::getName,
            R"(Returns the name that identifies the process variable.

            :return: The register name.
            :rtype: str)")
        .def("getUnit", &PyOneDRegisterAccessor::getUnit,
            R"(Returns the engineering unit.

            If none was specified, it will default to 'n./a.'.

            :return: The engineering unit string.
            :rtype: str)")
        .def("getDescription", &PyOneDRegisterAccessor::getDescription,
            R"(Returns the description of this variable/register.

            :return: The description string.
            :rtype: str)")
        .def("getValueType", &PyOneDRegisterAccessor::getValueType,
            R"(Returns the numpy dtype for the value type of this accessor.

            This can be used to determine the type at runtime.

            :return: Type information object.
            :rtype: numpy.dtype)")
        .def("getVersionNumber", &PyOneDRegisterAccessor::getVersionNumber,
            R"(Returns the version number that is associated with the last transfer.

            This refers to the last read or write operation.

            :return: The version number of the last transfer.
            :rtype: VersionNumber)")
        .def("isReadOnly", &PyOneDRegisterAccessor::isReadOnly,
            R"(Check if accessor is read only.

            This means it is readable but not writeable.

            :return: True if read only, false otherwise.
            :rtype: bool)")
        .def("isReadable", &PyOneDRegisterAccessor::isReadable,
            R"(Check if accessor is readable.

            :return: True if readable, false otherwise.
            :rtype: bool)")
        .def("isWriteable", &PyOneDRegisterAccessor::isWriteable,
            R"(Check if accessor is writeable.

            :return: True if writeable, false otherwise.
            :rtype: bool)")
        .def("getId", &PyOneDRegisterAccessor::getId,
            R"(Obtain unique ID for the actual implementation of this accessor.

            This means that e.g. two instances of OneDRegisterAccessor created by the same call to
            Device.getOneDRegisterAccessor() will have the same ID, while two instances obtained by two
            different calls to Device.getOneDRegisterAccessor() will have a different ID even when accessing
            the very same register.

            :return: The unique accessor ID.
            :rtype: TransferElementID)")
        .def("dataValidity", &PyOneDRegisterAccessor::dataValidity,
            R"(Return current validity of the data.

            Will always return DataValidity.ok if the backend does not support it.

            :return: The current data validity state.
            :rtype: DataValidity)")
        .def("getNElements", &PyOneDRegisterAccessor::getNElements,
            R"(Return number of elements/samples in the register.

            :return: Number of elements in the register.
            :rtype: int)")
        .def("get", &PyOneDRegisterAccessor::get,
            R"(Return the register data as an array (without a previous read).

            :return: Array containing the register data.
            :rtype: ndarray)")
        .def("set", &PyOneDRegisterAccessor::set, py::arg("newValue"),
            R"(Set the values of the array.

            :param newValue: New values to set in the buffer.
            :type newValue: list or ndarray)")
        .def("setAndWrite", &PyOneDRegisterAccessor::setAndWrite, py::arg("newValue"),
            py::arg("versionNumber") = PyVersionNumber::getNullVersion(),
            R"(Convenience function to set and write new value.

            If versionNumber is not specified, a new version number is generated.

            :param newValue: New values to set and write.
            :type newValue: list or ndarray
            :param versionNumber: Optional version number for the write operation.
            :type versionNumber: VersionNumber)")
        .def("getAsCooked", &PyOneDRegisterAccessor::getAsCooked, py::arg("element"),
            R"(Get the cooked values in case the accessor is a raw accessor (which does not do data conversion).

            This returns the converted data from the user buffer. It does not do any read or write transfer.

            :param element: Element index to read.
            :type element: int
            :return: The cooked value at the specified element.
            :rtype: int)")
        .def("setAsCooked", &PyOneDRegisterAccessor::setAsCooked, py::arg("element"), py::arg("value"),
            R"(Set the cooked values in case the accessor is a raw accessor (which does not do data conversion).

            This converts to raw and writes the data to the user buffer. It does not do any read or write transfer.

            :param element: Element index to write.
            :type element: int
            :param value: The cooked value to set.
            :type value: float)")
        .def("isInitialised", &PyOneDRegisterAccessor::isInitialised,
            R"(Check if the accessor is initialised.

            :return: True if initialised, false otherwise.
            :rtype: bool)")
        .def("setDataValidity", &PyOneDRegisterAccessor::setDataValidity, py::arg("validity"),
            R"(Set the data validity of the accessor.

            :param validity: The data validity state to set.
            :type validity: DataValidity)")
        .def("getAccessModeFlags", &PyOneDRegisterAccessor::getAccessModeFlags,
            R"(Return the access mode flags that were used to create this accessor.

            This can be used to determine the setting of the raw and the wait_for_new_data flags.

            :return: List of access mode flags.
            :rtype: list[AccessMode])")
        .def("readAndGet", &PyOneDRegisterAccessor::readAndGet,
            R"(Convenience function to read and return the register data.

            :return: Array containing the register data after reading.
            :rtype: ndarray)")
        .def("__getitem__", &PyOneDRegisterAccessor::getitem, py::arg("index"),
            R"(Get an element from the array by index.

            :param index: The element index.
            :type index: int
            :return: The value at the specified index.
            :rtype: scalar)")
        .def("__getattr__", &PyOneDRegisterAccessor::getattr);

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
