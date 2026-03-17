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
    py::class_<ctk::TransferGroup>(m, "TransferGroup")
        .def(py::init<>())
        .def(
            "addAccessor",
            [](ctk::TransferGroup& self, PyTransferElementBase& element) { self.addAccessor(element.getTE()); },
            R"(Add register to group. A register cannot be added to multiple groups. A TransferGroup can only be used with transfer elements that don't have AccessMode::wait_for_new_data.)")
        .def("read", &ctk::TransferGroup::read, R"(Trigger read transfer for all accessors in the group.)")
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
            R"(Trigger write transfer for all accessors in the group.)")
        .def("isReadOnly", &ctk::TransferGroup::isReadOnly,
            R"(Returns true if all accessors in the group are read only.)")
        .def("isReadable", &ctk::TransferGroup::isReadable,
            R"(Returns true if all accessors in the group are readable.)")
        .def("isWriteable", &ctk::TransferGroup::isWriteable,
            R"(Returns true if all accessors in the group are writable.)");
  }

  /********************************************************************************************************************/

} // namespace ChimeraTK
