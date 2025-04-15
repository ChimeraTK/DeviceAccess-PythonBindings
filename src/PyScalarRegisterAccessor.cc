// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyScalarRegisterAccessor.h"

#include <pybind11/stl.h>

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/

  void PyScalarRegisterAccessor::set(const UserTypeVariantNoVoid& val) {
    std::visit(
        [&](auto& acc) {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using expectedUserType = typename ACC::value_type;
          std::visit(
              [&](auto value) { acc.writeIfDifferent(ChimeraTK::userTypeToUserType<expectedUserType>(value)); }, val);
        },
        _accessor);
  }
  /********************************************************************************************************************/

  py::object PyScalarRegisterAccessor::readAndGet() {
    read();
    return get();
  }

  /********************************************************************************************************************/

  void PyScalarRegisterAccessor::setAndWrite(const UserTypeVariantNoVoid& val) {
    std::visit(
        [&](auto& acc) {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using expectedUserType = typename ACC::value_type;
          std::visit([&](auto value) { acc.setAndWrite(ChimeraTK::userTypeToUserType<expectedUserType>(value)); }, val);
        },
        _accessor);
  }
  /********************************************************************************************************************/

  py::object PyScalarRegisterAccessor::get() const {
    py::object rv;
    std::visit(
        [&](auto& acc) {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using userType = typename ACC::value_type;
          auto ndacc = boost::dynamic_pointer_cast<NDRegisterAccessor<userType>>(acc.getHighLevelImplElement());
          if constexpr(std::is_same<userType, std::string>::value) {
            // String arrays are not really supported by numpy, so we return a list instead
            rv = py::cast(ndacc->accessChannel(0));
          }
          else {
            auto ary = py::array(
                py::dtype::of<userType>(), {1}, {sizeof(userType)}, ndacc->accessChannel(0).data(), py::cast(this));
            assert(!ary.owndata()); // numpy must not own our buffers
            rv = ary;
          }
        },
        _accessor);
    return rv;
  }

  /********************************************************************************************************************/

  void PyScalarRegisterAccessor::writeIfDifferent(UserTypeVariantNoVoid val) {
    std::visit(
        [&](auto& acc) {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using expectedUserType = typename ACC::value_type;
          std::visit(
              [&](auto value) { acc.writeIfDifferent(ChimeraTK::userTypeToUserType<expectedUserType>(value)); }, val);
        },
        _accessor);
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
    py::class_<PyScalarRegisterAccessor, PyTransferElementBase> arrayacc(
        m, "ScalarRegisterAccessor", py::buffer_protocol());
    arrayacc.def(py::init<>())
        .def("read", &PyScalarRegisterAccessor::read,
            "Read the data from the device.\n\nIf AccessMode::wait_for_new_data was set, this function will block "
            "until new data has arrived. Otherwise it still might block for a short time until the data transfer was "
            "complete.")
        .def("readNonBlocking", &PyScalarRegisterAccessor::readNonBlocking,
            "Read the next value, if available in the input buffer.\n\nIf AccessMode::wait_for_new_data was set, this "
            "function returns immediately and the return value indicated if a new value was available (true) or not "
            "(false).\n\nIf AccessMode::wait_for_new_data was not set, this function is identical to read(), which "
            "will still return quickly. Depending on the actual transfer implementation, the backend might need to "
            "transfer data to obtain the current value before returning. Also this function is not guaranteed to be "
            "lock free. The return value will be always true in this mode.")
        .def("readLatest", &PyScalarRegisterAccessor::readLatest,
            "Read the latest value, discarding any other update since the last read if present.\n\nOtherwise this "
            "function is identical to readNonBlocking(), i.e. it will never wait for new values and it will return "
            "whether a new value was available if AccessMode::wait_for_new_data is set.")
        .def("write", &PyScalarRegisterAccessor::write,
            "Write the data to device.\n\nThe return value is true, old data was lost on the write transfer (e.g. due "
            "to an buffer overflow). In case of an unbuffered write transfer, the return value will always be false.")
        .def("writeDestructively", &PyScalarRegisterAccessor::writeDestructively,
            "Just like write(), but allows the implementation to destroy the content of the user buffer in the "
            "process.\n\nThis is an optional optimisation, hence there is a default implementation which just calls "
            "the normal doWriteTransfer(). In any case, the application must expect the user buffer of the "
            "TransferElement to contain undefined data after calling this function.")
        .def("getName", &PyScalarRegisterAccessor::getName, "Returns the name that identifies the process variable.")
        .def("getUnit", &PyScalarRegisterAccessor::getUnit,
            "Returns the engineering unit.\n\nIf none was specified, it will default to ' n./ a.'")
        .def("getDescription", &PyScalarRegisterAccessor::getDescription,
            "Returns the description of this variable/register.")
        .def("getValueType", &PyScalarRegisterAccessor::getValueType,
            "Returns the std::type_info for the value type of this transfer element.\n\nThis can be used to determine "
            "the type at runtime.")
        .def("getVersionNumber", &PyScalarRegisterAccessor::getVersionNumber,
            "Returns the version number that is associated with the last transfer (i.e. last read or write)")
        .def("isReadOnly", &PyScalarRegisterAccessor::isReadOnly,
            "Check if transfer element is read only, i.e. it is readable but not writeable.")
        .def("isReadable", &PyScalarRegisterAccessor::isReadable, "Check if transfer element is readable.")
        .def("isWriteable", &PyScalarRegisterAccessor::isWriteable, "Check if transfer element is writeable.")
        .def("getId", &PyScalarRegisterAccessor::getId,
            "Obtain unique ID for the actual implementation of this TransferElement.\n\nThis means that e.g. two "
            "instances of ScalarRegisterAccessor created by the same call to Device::getScalarRegisterAccessor() (e.g. "
            "by copying the accessor to another using NDRegisterAccessorBridge::replace()) will have the same ID, "
            "while two instances obtained by to difference calls to Device::getScalarRegisterAccessor() will have a "
            "different ID even when accessing the very same register.")
        .def("dataValidity", &PyScalarRegisterAccessor::dataValidity,
            "Return current validity of the data.\n\nWill always return DataValidity.ok if the backend does not "
            "support it")
        .def("get", &PyScalarRegisterAccessor::get, "Return a value of UserType (without a previous read).")
        .def("readAndGet", &PyScalarRegisterAccessor::readAndGet,
            "Convenience function to read and return a value of UserType.")
        .def("set", &PyScalarRegisterAccessor::set, "Set the value of UserType.", py::arg("val"))
        .def("write", &PyScalarRegisterAccessor::write,
            "Write the data to device.\n\nThe return value is true, old data was lost on the write transfer (e.g. due "
            "to an buffer overflow). In case of an unbuffered write transfer, the return value will always be false.")
        .def("writeDestructively", &PyScalarRegisterAccessor::writeDestructively,
            "Just like write(), but allows the implementation to destroy the content of the user buffer in the "
            "process.\n\nThis is an optional optimisation, hence there is a default implementation which just calls "
            "the normal doWriteTransfer(). In any case, the application must expect the user buffer of the "
            "TransferElement to contain undefined data after calling this function.")
        .def("writeIfDifferent", &PyScalarRegisterAccessor::writeIfDifferent,
            "Convenience function to set and write new value if it differes from the current value.\n\nThe given "
            "version number is only used in case the value differs. If versionNumber == {nullptr}, a new version "
            "number is generated only if the write actually takes place.",
            py::arg("val"))
        .def("setAndWrite", &PyScalarRegisterAccessor::setAndWrite,
            "Convenience function to set and write new value.\n\nThe given version number. If versionNumber == {}, a "
            "new version number is generated.",
            py::arg("val"))
        .def("__repr__", &PyScalarRegisterAccessor::repr);
    for(const auto& fn : PyTransferElementBase::specialFunctionsToEmulateNumeric) {
      arrayacc.def(fn.c_str(), [fn](PyScalarRegisterAccessor& acc, PyScalarRegisterAccessor& other) {
        return acc.get().attr(fn.c_str())(other.get());
      });
      arrayacc.def(fn.c_str(),
          [fn](PyScalarRegisterAccessor& acc, py::object& other) { return acc.get().attr(fn.c_str())(other); });
    }
    for(const auto& fn : PyTransferElementBase::specialUnaryFunctionsToEmulateNumeric) {
      arrayacc.def(fn.c_str(), [fn](PyScalarRegisterAccessor& acc) { return acc.get().attr(fn.c_str())(); });
    }
  }

  /********************************************************************************************************************/

} // namespace ChimeraTK