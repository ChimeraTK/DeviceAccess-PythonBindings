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
    std::string rep{"<OneDRegisterAccessor(type="};
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
    py::class_<PyOneDRegisterAccessor, PyTransferElementBase> arrayacc(
        m, "OneDRegisterAccessor", py::buffer_protocol());
    arrayacc.def(py::init<>())
        .def_buffer(&PyOneDRegisterAccessor::getBufferInfo)
        .def("read", &PyOneDRegisterAccessor::read,
            "Read the data from the device.\n\nIf AccessMode::wait_for_new_data was set, this function will block "
            "until new data has arrived. Otherwise it still might block for a short time until the data transfer was "
            "complete.")
        .def("readNonBlocking", &PyOneDRegisterAccessor::readNonBlocking,
            "Read the next value, if available in the input buffer.\n\nIf AccessMode::wait_for_new_data was set, this "
            "function returns immediately and the return value indicated if a new value was available (true) or not "
            "(false).\n\nIf AccessMode::wait_for_new_data was not set, this function is identical to read(), which "
            "will still return quickly. Depending on the actual transfer implementation, the backend might need to "
            "transfer data to obtain the current value before returning. Also this function is not guaranteed to be "
            "lock free. The return value will be always true in this mode.")
        .def("readLatest", &PyOneDRegisterAccessor::readLatest,
            "Read the latest value, discarding any other update since the last read if present.\n\nOtherwise this "
            "function is identical to readNonBlocking(), i.e. it will never wait for new values and it will return "
            "whether a new value was available if AccessMode::wait_for_new_data is set.")
        .def("write", &PyOneDRegisterAccessor::write,
            "Write the data to device.\n\nThe return value is true, old data was lost on the write transfer (e.g. due "
            "to an buffer overflow). In case of an unbuffered write transfer, the return value will always be false.",
            py::arg("versionNumber") = PyVersionNumber::getNullVersion())
        .def("writeDestructively", &PyOneDRegisterAccessor::writeDestructively,
            "Just like write(), but allows the implementation to destroy the content of the user buffer in the "
            "process.\n\nThis is an optional optimisation, hence there is a default implementation which just calls "
            "the normal doWriteTransfer(). In any case, the application must expect the user buffer of the "
            "TransferElement to contain undefined data after calling this function.",
            py::arg("versionNumber") = PyVersionNumber::getNullVersion())
        .def("interrupt", &PyOneDRegisterAccessor::interrupt,
            "Return from a blocking read immediately and throw the ThreadInterrupted exception.")
        .def("getName", &PyOneDRegisterAccessor::getName, "Returns the name that identifies the process variable.")
        .def("getUnit", &PyOneDRegisterAccessor::getUnit,
            "Returns the engineering unit.\n\nIf none was specified, it will default to ' n./ a.'")
        .def("getDescription", &PyOneDRegisterAccessor::getDescription,
            "Returns the description of this variable/register.")
        .def("getValueType", &PyOneDRegisterAccessor::getValueType,
            "Returns the std::type_info for the value type of this transfer element.\n\nThis can be used to determine "
            "the type at runtime.")
        .def("getVersionNumber", &PyOneDRegisterAccessor::getVersionNumber,
            "Returns the version number that is associated with the last transfer (i.e. last read or write)")
        .def("isReadOnly", &PyOneDRegisterAccessor::isReadOnly,
            "Check if transfer element is read only, i.e. it is readable but not writeable.")
        .def("isReadable", &PyOneDRegisterAccessor::isReadable, "Check if transfer element is readable.")
        .def("isWriteable", &PyOneDRegisterAccessor::isWriteable, "Check if transfer element is writeable.")
        .def("getId", &PyOneDRegisterAccessor::getId,
            "Obtain unique ID for the actual implementation of this TransferElement.\n\nThis means that e.g. two "
            "instances of ScalarRegisterAccessor created by the same call to Device::getScalarRegisterAccessor() (e.g. "
            "by copying the accessor to another using NDRegisterAccessorBridge::replace()) will have the same ID, "
            "while two instances obtained by to difference calls to Device::getScalarRegisterAccessor() will have a "
            "different ID even when accessing the very same register.")
        .def("dataValidity", &PyOneDRegisterAccessor::dataValidity,
            "Return current validity of the data.\n\nWill always return DataValidity.ok if the backend does not "
            "support it")
        .def(
            "getNElements", &PyOneDRegisterAccessor::getNElements, "Return number of elements/samples in the register.")
        .def("get", &PyOneDRegisterAccessor::get, "Return an array of UserType (without a previous read).")
        .def("set", &PyOneDRegisterAccessor::set, "Set the values of the array of UserType.", py::arg("newValue"))
        .def("setAndWrite", &PyOneDRegisterAccessor::setAndWrite,
            "Convenience function to set and write new value.\n\nThe given version number. If versionNumber == {}, a"
            "new version number is generated.",
            py::arg("newValue"), py::arg("versionNumber") = PyVersionNumber::getNullVersion())
        .def("getAsCooked", &PyOneDRegisterAccessor::getAsCooked,
            "Get the cooked values in case the accessor is a raw accessor (which does not do data conversion). This "
            "returns the converted data from the use buffer. It does not do any read or write transfer.",
            py::arg("element"))
        .def("setAsCooked", &PyOneDRegisterAccessor::setAsCooked,
            "Set the cooked values in case the accessor is a raw accessor (which does not do data conversion). This "
            "converts to raw and writes the data to the user buffer. It does not do any read or write transfer.",
            py::arg("element"), py::arg("value"))
        .def("isInitialised", &PyOneDRegisterAccessor::isInitialised, "Check if the transfer element is initialised.")
        .def("setDataValidity", &PyOneDRegisterAccessor::setDataValidity,
            "Set the data validity of the transfer element.")
        .def("getAccessModeFlags", &PyOneDRegisterAccessor::getAccessModeFlags,
            "Return the access mode flags that were used to create this TransferElement.\n\nThis can be used to "
            "determine the setting of the `raw` and the `wait_for_new_data` flags")
        .def("readAndGet", &PyOneDRegisterAccessor::readAndGet,
            "Convenience function to read and return an array of UserType.")
        .def("__getitem__", &PyOneDRegisterAccessor::getitem, "Get an element from the array by index.")
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
