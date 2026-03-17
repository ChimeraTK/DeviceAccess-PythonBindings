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
    py::class_<ctk::ReadAnyGroup>(m, "ReadAnyGroup")
        .def(py::init<>())
        .def("finalise", &ctk::ReadAnyGroup::finalise,
            R"(Finalise the group. After this, add() may no longer be called and read methods may be used.)")
        .def("interrupt", &ctk::ReadAnyGroup::interrupt, R"(Interrupt the group.)")
        .def(
            "add", [](ctk::ReadAnyGroup& self, PyTransferElementBase& element) { self.add(element.getTE()); },
            R"(Add register to group. Note that calling this function is only allowed before finalise() has been called. The given register may not yet be part of a ReadAnyGroup or a TransferGroup, otherwise an exception is thrown. The register must be must be readable.)",
            py::arg("element"))
        .def("readAny", &ctk::ReadAnyGroup::readAny,
            R"(Wait until one of the elements in this group has received an update.)")
        .def("readAnyNonBlocking", &ctk::ReadAnyGroup::readAnyNonBlocking,
            R"(Wait until one of the elements in this group has received an update.)")
        .def(
            "readUntil", [](ctk::ReadAnyGroup& self, const ctk::TransferElementID& id) { self.readUntil(id); },
            R"(Wait until the given TransferElementID has received an update and store it to its user buffer. )",
            py::arg("id"))
        .def(
            "readUntil",
            [](ctk::ReadAnyGroup& self, const PyTransferElementBase& element) { self.readUntil(element.getTE()); },
            R"(Wait until the given TransferElement has received an update and store it to its user buffer.)",
            py::arg("element"))
        .def(
            "readUntilAll",
            [](ctk::ReadAnyGroup& self, const std::vector<ctk::TransferElementID>& ids) { self.readUntilAll(ids); },
            R"(Wait until all of the given TransferElementID has received an update and store it to its user buffer.)",
            py::arg("ids"))
        .def(
            "readUntilAll",
            [](ctk::ReadAnyGroup& self, const py::list& elements) {
              // implementation from deviceaccess/src/ReadAnyGroup.cpp, adapted to work with PyTransferElementBase
              // wihtout to template every possible usertype
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
            R"(Wait until all of the given TransferElement has received an update and store it to its user buffer.)",
            py::arg("element"))
        .def("waitAny", &ctk::ReadAnyGroup::waitAny,
            R"(Wait until any of the elements in this group has received an update, but do not process the update.)")
        .def("waitAnyNonBlocking", &ctk::ReadAnyGroup::waitAnyNonBlocking,
            R"(Check if an update is available in the group, but do not block if no update is available.)")
        .def("processPolled", &ctk::ReadAnyGroup::processPolled,
            R"(Process polled transfer elements (update them if new values are available.)");
  }

  /*******************************************************************************************************************/

  void PyReadAnyGroupNotification::bind(py::module& m) {
    py::class_<ctk::ReadAnyGroup::Notification>(m, "Notification")
        .def(py::init<>())
        .def("accept", &ctk::ReadAnyGroup::Notification::accept, R"(Accept the notification.)")
        .def("getId", &ctk::ReadAnyGroup::Notification::getId,
            R"(Return the ID of the transfer element for which this notification has been generated.)")
        .def(
            "getTransferElement",
            [](ctk::ReadAnyGroup::Notification&) {
              // Implementation would need a different architecture, as we cannot just return ther cpp transfer element.
              // We would need to change the ReadAnyGroup to keep references to the added accessors and return those here.
              throw std::runtime_error(
                  "getTransferElement() is not implemented yet, please contact the developers if you need this.");
            },
            R"(Return the transfer element for which this notification has been generated.)")
        .def("isReady", &ctk::ReadAnyGroup::Notification::isReady,
            R"(Tell whether this notification is valid and has not been accepted yet.)");
  }

  /********************************************************************************************************************/

} // namespace ChimeraTK
