// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyTransferGroup.h"

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

  void PyTransferGroup::bind(py::module& m) {
    py::class_<ctk::TransferGroup>(m, "TransferGroup",
        R"(Group of TransferElements for coordinated read and write operations.

      A TransferGroup allows triggering one read or write call for all added accessors.)")
        .def(py::init<>(), R"(Create an empty TransferGroup.)")
        .def(
            "addAccessor",
            [](ctk::TransferGroup& self, PyTransferElementBase& element) { self.addAccessor(element.getTE()); },
            py::arg("element"),
            R"(Add an accessor to the group.

        A TransferElement can only be part of one TransferGroup. TransferGroup can only be used with transfer
        elements that do not have AccessMode.wait_for_new_data.

        Args:
          element (TransferElementBase): Accessor to add to the group.)")
        .def("read", &ctk::TransferGroup::read, R"(Trigger a read transfer for all accessors in the group.)")
        .def(
            "write",
            [](ctk::TransferGroup& self, PyVersionNumber versionNumber) {
              if(versionNumber == ChimeraTK::VersionNumber{nullptr}) {
                self.write();
              }
              else {
                self.write(versionNumber);
              }
            },
            py::arg("versionNumber") = PyVersionNumber::getNullVersion(),
            R"(Trigger a write transfer for all accessors in the group.

            Args:
              versionNumber (VersionNumber): Optional version number used for the write operation. If not set, a new
              version number is generated.)")
        .def("isReadOnly", &ctk::TransferGroup::isReadOnly,
            R"(Check whether all accessors in the group are read only.

            Returns:
              bool: True if all accessors are read only, false otherwise.)")
        .def("isReadable", &ctk::TransferGroup::isReadable,
            R"(Check whether all accessors in the group are readable.

            Returns:
              bool: True if all accessors are readable, false otherwise.)")
        .def("isWriteable", &ctk::TransferGroup::isWriteable,
            R"(Check whether all accessors in the group are writable.

            Returns:
              bool: True if all accessors are writable, false otherwise.)");
  }

  /********************************************************************************************************************/

} // namespace ChimeraTK
