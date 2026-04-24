// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyReadAnyGroup.h"

#include "PyTransferElement.h"

#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
namespace ctk = ChimeraTK;

namespace py = pybind11;
using namespace pybind11::literals;

namespace ChimeraTK {

  /********************************************************************************************************************/

  void PyReadAnyGroup::bind(py::module& m) {
    py::class_<ctk::ReadAnyGroup>(m, "ReadAnyGroup",
        R"(Group for waiting on updates from multiple TransferElements with wait_for_new_data enabled.)")
        .def(py::init<>(), R"(Create an empty ReadAnyGroup.)")
        .def("finalise", &ctk::ReadAnyGroup::finalise,
            R"(Finalise the group.

            After this, add() may no longer be called and read/wait methods may be used.

            Returns:
                None: This function does not return a value.)")
        .def("interrupt", &ctk::ReadAnyGroup::interrupt, R"(Interrupt blocking operations running on this group.)")
        .def(
            "add", [](ctk::ReadAnyGroup& self, PyTransferElementBase& element) { self.add(element.getTE()); },
            py::arg("element"),
            R"(Add a TransferElement to the group.

            This is only allowed before finalise() has been called. The given element may not already be part of a
            ReadAnyGroup or TransferGroup, otherwise an exception is thrown. The element must be readable.

            Returns:
                None: This function does not return a value.)")
        .def("readAny", &ctk::ReadAnyGroup::readAny,
            R"(Wait until one of the elements in this group has received an update.

            Returns:
                TransferElementID: ID of the element whose update has been processed.)")
        .def("readAnyNonBlocking", &ctk::ReadAnyGroup::readAnyNonBlocking,
            R"(Return immediately and process an update if one is available.

            Returns:
                TransferElementID: ID of the processed element, or an invalid ID if no update is available.)")
        .def(
            "readUntil", [](ctk::ReadAnyGroup& self, const ctk::TransferElementID& id) { self.readUntil(id); },
            py::arg("id"),
            R"(Wait until the given TransferElementID has received an update and store it in its user buffer.

            Returns:
                None: This function does not return a value.)")
        .def(
            "readUntil",
            [](ctk::ReadAnyGroup& self, const PyTransferElementBase& element) { self.readUntil(element.getTE()); },
            py::arg("element"),
            R"(Wait until the given TransferElement has received an update and store it in its user buffer.

            Returns:
                None: This function does not return a value.)")
        .def(
            "readUntilAll",
            [](ctk::ReadAnyGroup& self, const std::vector<ctk::TransferElementID>& ids) { self.readUntilAll(ids); },
            py::arg("ids"),
            R"(Wait until all given TransferElementID values have received updates and store them in their user buffers.

            Returns:
                None: This function does not return a value.)")
        .def(
            "readUntilAll",
            [](ctk::ReadAnyGroup& self, const py::list& elements) {
              // implementation from deviceaccess/src/ReadAnyGroup.cpp, adapted to work with PyTransferElementBase
              // without templating every possible user type
              auto locals = py::dict("self"_a = self, "elements"_a = elements);
              py::exec(R"(
                found = {}
                for elem in elements:
                    found[elem.getId()] = False
                leftToFind = len(elements)
                while True:
                    read = self.readAny()
                    if read not in found:
                        continue
                    if found[read]:
                        continue
                    found[read] = True
                    leftToFind -= 1
                    if leftToFind == 0:
                        break
              )",
                  py::globals(), locals);
            },
            py::arg("elements"),
            R"(Wait until all given TransferElements have received updates and store them in their user buffers.

            Returns:
                None: This function does not return a value.)")
        .def("waitAny", &ctk::ReadAnyGroup::waitAny,
            R"(Wait until any element in this group has received an update, but do not process the update.

            Returns:
                Notification: Notification object for the pending update.)")
        .def("waitAnyNonBlocking", &ctk::ReadAnyGroup::waitAnyNonBlocking,
            R"(Return immediately with a notification if an update is available.

            Returns:
                Notification: Notification object for a pending update, or an invalid notification if none is available.)")
        .def("processPolled", &ctk::ReadAnyGroup::processPolled,
            R"(Process polled TransferElements and update them if new values are available.

            Returns:
                None: This function does not return a value.)");
  }

  /*******************************************************************************************************************/

  void PyReadAnyGroupNotification::bind(py::module& m) {
    py::class_<ctk::ReadAnyGroup::Notification>(
        m, "Notification", R"(Notification returned by ReadAnyGroup wait methods.)")
        .def(py::init<>(), R"(Create an invalid notification.)")
        .def("accept", &ctk::ReadAnyGroup::Notification::accept,
            R"(Accept the notification and process the associated update.

            Returns:
                None: This function does not return a value.)")
        .def("getId", &ctk::ReadAnyGroup::Notification::getId,
            R"(Return the ID of the TransferElement for which this notification was generated.

            Returns:
                TransferElementID: ID of the associated TransferElement.)")
        .def(
            "getTransferElement",
            [](ctk::ReadAnyGroup::Notification&) {
              // Implementation would need a different architecture, as we cannot just return the C++ TransferElement.
              // We would need to change the ReadAnyGroup to keep references to the added accessors and return those here.
              throw std::runtime_error(
                  "getTransferElement() is not implemented yet, please contact the developers if you need this.");
            },
            R"(Return the TransferElement for which this notification was generated.

            Raises:
                RuntimeError: Always raised because this method is not implemented.)")
        .def("isReady", &ctk::ReadAnyGroup::Notification::isReady,
            R"(Tell whether this notification is valid and has not been accepted yet.

            Returns:
                bool: True if this notification is ready to be accepted, false otherwise.)");
  }

  /********************************************************************************************************************/

} // namespace ChimeraTK
