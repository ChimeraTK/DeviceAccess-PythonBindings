// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyVoidRegisterAccessor.h"

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

  void PyVoidRegisterAccessor::bind(py::module& m) {
    py::class_<PyVoidRegisterAccessor>(m, "VoidRegisterAccessor")
        .def(py::init<>())
        .def("read", &ChimeraTK::VoidRegisterAccessor::read, py::call_guard<py::gil_scoped_release>(),
            "Read the data from the device.\n\nIf AccessMode::wait_for_new_data was set, this function will block "
            "until new data has arrived. Otherwise it still might block for a short time until the data transfer was "
            "complete.")
        .def("readNonBlocking", &ChimeraTK::VoidRegisterAccessor::readNonBlocking,
            py::call_guard<py::gil_scoped_release>(),
            "Read the next value, if available in the input buffer.\n\nIf AccessMode::wait_for_new_data was set, this "
            "function returns immediately and the return value indicated if a new value was available (true) or not "
            "(false).\n\nIf AccessMode::wait_for_new_data was not set, this function is identical to read(), which "
            "will still return quickly. Depending on the actual transfer implementation, the backend might need to "
            "transfer data to obtain the current value before returning. Also this function is not guaranteed to be "
            "lock free. The return value will be always true in this mode.")
        .def("readLatest", &ChimeraTK::VoidRegisterAccessor::readLatest, py::call_guard<py::gil_scoped_release>(),
            "Read the latest value, discarding any other update since the last read if present.\n\nOtherwise this "
            "function is identical to readNonBlocking(), i.e. it will never wait for new values and it will return "
            "whether a new value was available if AccessMode::wait_for_new_data is set.")
        .def("write", &ChimeraTK::VoidRegisterAccessor::write, py::call_guard<py::gil_scoped_release>(),
            "Write the data to device.\n\nThe return value is true, old data was lost on the write transfer (e.g. due "
            "to an buffer overflow). In case of an unbuffered write transfer, the return value will always be false.")
        .def("writeDestructively", &ChimeraTK::VoidRegisterAccessor::writeDestructively,
            py::call_guard<py::gil_scoped_release>(),
            "Just like write(), but allows the implementation to destroy the content of the user buffer in the "
            "process.\n\nThis is an optional optimisation, hence there is a default implementation which just calls "
            "the normal doWriteTransfer(). In any case, the application must expect the user buffer of the "
            "TransferElement to contain undefined data after calling this function.")
        .def("getName", &ChimeraTK::VoidRegisterAccessor::getName, py::call_guard<py::gil_scoped_release>(),
            "Returns the name that identifies the process variable.")
        .def("getUnit", &ChimeraTK::VoidRegisterAccessor::getUnit, py::call_guard<py::gil_scoped_release>(),
            "Returns the engineering unit.\n\nIf none was specified, it will default to ' n./ a.'")
        .def("getDescription", &ChimeraTK::VoidRegisterAccessor::getDescription,
            py::call_guard<py::gil_scoped_release>(), "Returns the description of this variable/register.")
        .def("getValueType", &ChimeraTK::VoidRegisterAccessor::getValueType, py::call_guard<py::gil_scoped_release>(),
            "Returns the std::type_info for the value type of this transfer element.\n\nThis can be used to determine "
            "the type at runtime.")
        .def("getVersionNumber", &ChimeraTK::VoidRegisterAccessor::getVersionNumber,
            py::call_guard<py::gil_scoped_release>(),
            "Returns the version number that is associated with the last transfer (i.e. last read or write)")
        .def("isReadOnly", &ChimeraTK::VoidRegisterAccessor::isReadOnly, py::call_guard<py::gil_scoped_release>(),
            "Check if transfer element is read only, i.e. it is readable but not writeable.")
        .def("isReadable", &ChimeraTK::VoidRegisterAccessor::isReadable, py::call_guard<py::gil_scoped_release>(),
            "Check if transfer element is readable.")
        .def("isWriteable", &ChimeraTK::VoidRegisterAccessor::isWriteable, py::call_guard<py::gil_scoped_release>(),
            "Check if transfer element is writeable.")
        .def("getId", &ChimeraTK::VoidRegisterAccessor::getId, py::call_guard<py::gil_scoped_release>(),
            "Obtain unique ID for the actual implementation of this TransferElement.\n\nThis means that e.g. two "
            "instances of ScalarRegisterAccessor created by the same call to Device::getScalarRegisterAccessor() (e.g. "
            "by copying the accessor to another using NDRegisterAccessorBridge::replace()) will have the same ID, "
            "while two instances obtained by to difference calls to Device::getScalarRegisterAccessor() will have a "
            "different ID even when accessing the very same register.")
        .def("dataValidity", &ChimeraTK::VoidRegisterAccessor::dataValidity, py::call_guard<py::gil_scoped_release>(),
            "Return current validity of the data.\n\nWill always return DataValidity.ok if the backend does not "
            "support it");
  }

  /********************************************************************************************************************/

} // namespace ChimeraTK