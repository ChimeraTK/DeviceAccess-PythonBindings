// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyVoidRegisterAccessor.h"

#include "PyVersionNumber.h"

#include <ChimeraTK/VersionNumber.h>

#include <pybind11/stl.h>

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/

  std::string PyVoidRegisterAccessor::repr(py::object& acc) const {
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

  bool PyVoidRegisterAccessor::write(const PyVersionNumber& versionNumber) {
    if(versionNumber == PyVersionNumber::getNullVersion()) {
      // The pybind11 default argument is set during compile time, so we use the null version to mark the default and
      // act like nothing was set.
      return ChimeraTK::VoidRegisterAccessor::write();
    }
    return ChimeraTK::VoidRegisterAccessor::write(versionNumber);
  }

  /********************************************************************************************************************/

  bool PyVoidRegisterAccessor::writeDestructively(const PyVersionNumber& versionNumber) {
    if(versionNumber == PyVersionNumber::getNullVersion()) {
      // The pybind11 default argument is set during compile time, so we use the null version to mark the default and
      // act like nothing was set.
      return ChimeraTK::VoidRegisterAccessor::writeDestructively();
    }
    return ChimeraTK::VoidRegisterAccessor::writeDestructively(versionNumber);
  }

  /********************************************************************************************************************/

  void PyVoidRegisterAccessor::bind(py::module& m) {
    py::class_<PyVoidRegisterAccessor>(m, "VoidRegisterAccessor")
        .def(py::init<>())
        .def("read", &ChimeraTK::VoidRegisterAccessor::read,
            "Read the data from the device.\n\nIf AccessMode::wait_for_new_data was set, this function will block "
            "until new data has arrived. Otherwise it still might block for a short time until the data transfer was "
            "complete.",
            py::call_guard<py::gil_scoped_release>())
        .def("readNonBlocking", &ChimeraTK::VoidRegisterAccessor::readNonBlocking,

            "Read the next value, if available in the input buffer.\n\nIf AccessMode::wait_for_new_data was set, this "
            "function returns immediately and the return value indicated if a new value was available (true) or not "
            "(false).\n\nIf AccessMode::wait_for_new_data was not set, this function is identical to read(), which "
            "will still return quickly. Depending on the actual transfer implementation, the backend might need to "
            "transfer data to obtain the current value before returning. Also this function is not guaranteed to be "
            "lock free. The return value will be always true in this mode.")
        .def("readLatest", &ChimeraTK::VoidRegisterAccessor::readLatest,
            "Read the latest value, discarding any other update since the last read if present.\n\nOtherwise this "
            "function is identical to readNonBlocking(), i.e. it will never wait for new values and it will return "
            "whether a new value was available if AccessMode::wait_for_new_data is set.")
        .def("write", &PyVoidRegisterAccessor::write,
            "Write the data to device.\n\nThe return value is true, old data was lost on the write transfer (e.g. due "
            "to an buffer overflow). In case of an unbuffered write transfer, the return value will always be false.",
            py::arg("versionNumber") = PyVersionNumber::getNullVersion())
        .def("writeDestructively", &PyVoidRegisterAccessor::writeDestructively,
            "Just like write(), but allows the implementation to destroy the content of the user buffer in the "
            "process.\n\nThis is an optional optimisation, hence there is a default implementation which just calls "
            "the normal doWriteTransfer(). In any case, the application must expect the user buffer of the "
            "TransferElement to contain undefined data after calling this function.",
            py::arg("versionNumber") = PyVersionNumber::getNullVersion())
        .def("interrupt", &ChimeraTK::VoidRegisterAccessor::interrupt,
            "Return from a blocking read immediately and throw the ThreadInterrupted exception.")
        .def("getName", &ChimeraTK::VoidRegisterAccessor::getName,
            "Returns the name that identifies the process variable.")
        .def("getUnit", &ChimeraTK::VoidRegisterAccessor::getUnit,
            "Returns the engineering unit.\n\nIf none was specified, it will default to ' n./ a.'")
        .def("getDescription", &ChimeraTK::VoidRegisterAccessor::getDescription,
            "Returns the description of this variable/register.")
        .def("getValueType", &ChimeraTK::VoidRegisterAccessor::getValueType,
            "Returns the std::type_info for the value type of this transfer element.\n\nThis can be used to determine "
            "the type at runtime.")
        .def("setDataValidity", &PyVoidRegisterAccessor::setDataValidity,
            "Set the data validity of the transfer element.")
        .def("isInitialised", &ChimeraTK::VoidRegisterAccessor::isInitialised,
            "Check if the transfer element is initialised.")
        .def("getVersionNumber", &ChimeraTK::VoidRegisterAccessor::getVersionNumber,
            "Returns the version number that is associated with the last transfer (i.e. last read or write)")
        .def("isReadOnly", &ChimeraTK::VoidRegisterAccessor::isReadOnly,
            "Check if transfer element is read only, i.e. it is readable but not writeable.")
        .def("isReadable", &ChimeraTK::VoidRegisterAccessor::isReadable, "Check if transfer element is readable.")
        .def("isWriteable", &ChimeraTK::VoidRegisterAccessor::isWriteable, "Check if transfer element is writeable.")
        .def("getId", &ChimeraTK::VoidRegisterAccessor::getId,
            "Obtain unique ID for the actual implementation of this TransferElement.\n\nThis means that e.g. two "
            "instances of ScalarRegisterAccessor created by the same call to Device::getScalarRegisterAccessor() (e.g. "
            "by copying the accessor to another using NDRegisterAccessorBridge::replace()) will have the same ID, "
            "while two instances obtained by to difference calls to Device::getScalarRegisterAccessor() will have a "
            "different ID even when accessing the very same register.")
        .def("getAccessModeFlags", &PyVoidRegisterAccessor::getAccessModeFlagsAsList,
            "Return the access mode flags that were used to create this TransferElement.\n\nThis can be used to "
            "determine the setting of the `raw` and the `wait_for_new_data` flags")
        .def("dataValidity", &ChimeraTK::VoidRegisterAccessor::dataValidity,
            "Return current validity of the data.\n\nWill always return DataValidity.ok if the backend does not "
            "support it");
  }

  /********************************************************************************************************************/

} // namespace ChimeraTK
