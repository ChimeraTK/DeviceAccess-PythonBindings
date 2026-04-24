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
    std::string rep{"<PyVoidRegisterAccessor(type="};
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
    py::class_<PyVoidRegisterAccessor>(m, "VoidRegisterAccessor",
        R"(Special accessor that represents a register with no user data (ChimeraTK::Void).

        This accessor is typically used to model triggers or actions that do not carry a payload. There is no user
        buffer to read or write values from; read()/write() only perform transfer synchronisation and version-tagged
        signalling.

        Note:
            Create instances via Device.getVoidRegisterAccessor(). There is no get()/set() as there is no value to
            access. Use write()/writeDestructively() to trigger an action and read()/readNonBlocking()/readLatest() to
            synchronise with incoming events.)")
        .def(py::init<>())
        .def("read", &ChimeraTK::VoidRegisterAccessor::read,
            R"(Read from the device to synchronise with the latest event.

            If AccessMode.wait_for_new_data was set, this function will block until new data has arrived. Otherwise it
            still might block for a short time until the data transfer was complete.

            See Also:
                readNonBlocking: Read without blocking if no data is available.
                readLatest: Read latest value while discarding intermediate updates.

            Returns:
                None: This function does not return a value.)",
            py::call_guard<py::gil_scoped_release>())
        .def("readNonBlocking", &ChimeraTK::VoidRegisterAccessor::readNonBlocking,
            R"(Read the next event, if available.

            If AccessMode.wait_for_new_data was set, this function returns immediately and the return value indicates
            if a new value was available (true) or not (false).

            If AccessMode.wait_for_new_data was not set, this function is identical to read(), which will still return
            quickly. Depending on the actual transfer implementation, the backend might need to transfer data to obtain
            the current state before returning. Also this function is not guaranteed to be lock free. The return value
            will be always true in this mode.

            Returns:
                bool: True if new data was available, false otherwise.)")
        .def("readLatest", &ChimeraTK::VoidRegisterAccessor::readLatest,
            R"(Read the latest event, discarding intermediate updates since the last read if present.

            Otherwise this function is identical to readNonBlocking(), i.e. it will never wait for new values and it
            will return whether a new value was available if AccessMode.wait_for_new_data is set.

            Returns:
                bool: True if new data was available, false otherwise.)")
        .def("write", &PyVoidRegisterAccessor::write,
            R"(Trigger a write to the device (no payload).

            The return value is true if old data was lost on the write transfer (e.g. due to a buffer overflow). In
            case of an unbuffered write transfer, the return value will always be false.

            Args:
                versionNumber (VersionNumber): Version number to use for this write operation. If not specified, a new
                    version number is generated.

            Returns:
                bool: True if old data was lost on the write transfer.

            See Also:
                writeDestructively: Optimized write that may destroy buffer.)",
            py::arg("versionNumber") = PyVersionNumber::getNullVersion())
        .def("writeDestructively", &PyVoidRegisterAccessor::writeDestructively,
            R"(Like write(), but allows the implementation to destroy the content of internal buffers in the process.

            This is an optional optimisation, hence there is a default implementation which just calls write(). In any
            case, the application must expect internal buffers to contain undefined data after calling this function.

            Args:
                versionNumber (VersionNumber): Version number to use for this write operation. If not specified, a new
                    version number is generated.

            Returns:
                bool: True if old data was lost on the write transfer.)",
            py::arg("versionNumber") = PyVersionNumber::getNullVersion())
        .def("interrupt", &ChimeraTK::VoidRegisterAccessor::interrupt,
            R"(Interrupt a blocking read operation.

            This will cause a blocking read to return immediately and throw an InterruptedException.

            Returns:
                None: This function does not return a value.)")
        .def("getName", &ChimeraTK::VoidRegisterAccessor::getName,
            R"(Return the name that identifies the process variable.

            Returns:
                str: The register name.)")
        .def("getUnit", &ChimeraTK::VoidRegisterAccessor::getUnit,
            R"(Return the engineering unit.

            If none was specified, it will default to 'n./a.'.

            Returns:
                str: The engineering unit string.)")
        .def("getDescription", &ChimeraTK::VoidRegisterAccessor::getDescription,
            R"(Return the description of this variable/register.

            Returns:
                str: The description string.)")
        .def("getValueType", &ChimeraTK::VoidRegisterAccessor::getValueType,
            R"(Return the type_info for the value type of this accessor.

            This can be used to determine the type at runtime.

            Returns:
                type: Type information object.)")
        .def("setDataValidity", &PyVoidRegisterAccessor::setDataValidity, py::arg("validity"),
            R"(Set the data validity of the accessor.

            Args:
                validity (DataValidity): The data validity state to set.

            Returns:
                None: This function does not return a value.)")
        .def("isInitialised", &ChimeraTK::VoidRegisterAccessor::isInitialised,
            R"(Check if the accessor is initialised.

            Returns:
                bool: True if initialised, false otherwise.)")
        .def("getVersionNumber", &ChimeraTK::VoidRegisterAccessor::getVersionNumber,
            R"(Return the version number that is associated with the last transfer.

            This refers to the last read or write operation.

            Returns:
                VersionNumber: The version number of the last transfer.)")
        .def("isReadOnly", &ChimeraTK::VoidRegisterAccessor::isReadOnly,
            R"(Check if accessor is read only.

            This means it is readable but not writeable.

            Returns:
                bool: True if read only, false otherwise.)")
        .def("isReadable", &ChimeraTK::VoidRegisterAccessor::isReadable,
            R"(Check if accessor is readable.

            Returns:
                bool: True if readable, false otherwise.)")
        .def("isWriteable", &ChimeraTK::VoidRegisterAccessor::isWriteable,
            R"(Check if accessor is writeable.

            Returns:
                bool: True if writeable, false otherwise.)")
        .def("getId", &ChimeraTK::VoidRegisterAccessor::getId,
            R"(Obtain unique ID for the actual implementation of this accessor.

            This means that e.g. two instances created by the same call to Device.getVoidRegisterAccessor() will have
            the same ID, while two instances obtained by two different calls will have a different ID even when
            accessing the very same register.

            Returns:
                TransferElementID: The unique TransferElement ID.)")
        .def("getAccessModeFlags", &PyVoidRegisterAccessor::getAccessModeFlagsAsList,
            R"(Return the access mode flags that were used to create this accessor.

            This can be used to determine the setting of the raw and the wait_for_new_data flags.

            Returns:
                list[AccessMode]: List of access mode flags.)")
        .def("dataValidity", &ChimeraTK::VoidRegisterAccessor::dataValidity,
            R"(Return current validity of the data.

            Will always return DataValidity.ok if the backend does not support it.

            Returns:
                DataValidity: The current data validity state.)");
  }

  /********************************************************************************************************************/

} // namespace ChimeraTK
