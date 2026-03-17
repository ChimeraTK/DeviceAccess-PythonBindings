// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyDataConsistencyGroup.h"

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

  void PyDataConsistencyGroup::bind(py::module& m) {
    py::class_<ctk::DataConsistencyGroup>(m, "DataConsistencyGroup")
        .def(py::init<ctk::DataConsistencyGroup::MatchingMode>())
        .def(
            "add", [](ctk::DataConsistencyGroup& self, PyTransferElementBase& element) { self.add(element.getTE()); },
            R"(Add register to group.
            The same TransferElement can be part of multiple DataConsistencyGroups. The register must be must be
    readable, and it must have AccessMode::wait_for_new_data.)")
        .def("update", &ctk::DataConsistencyGroup::update,
            R"(This function must be called after an update was received from the ReadAnyGroup.
            It returns true, if a consistent state is reached. It returns false if an TransferElementID was updated, that was not added to this group. For MatchingMode::historized, readAny will only let through consistent updates, so then update always returns true.)")
        .def("getMatchingMode", &ctk::DataConsistencyGroup::getMatchingMode,
            R"(Get the current MatchingMode of this DataConsistencyGroup.)")
        .def("isConsistent", &ctk::DataConsistencyGroup::isConsistent,
            R"(Returns true if a consistent state is reached )");
  }

  /*******************************************************************************************************************/

  void PyMatchingMode::bind(py::module& m) {
    py::enum_<ctk::DataConsistencyGroup::MatchingMode>(
        m, "MatchingMode", "Enum describing the matching mode of a DataConsistencyGroup.")
        .value("none", ctk::DataConsistencyGroup::MatchingMode::none,
            "No matching, effectively disable the DataConsitencyGroup. update() will always return true. ")
        .value("exact", ctk::DataConsistencyGroup::MatchingMode::exact,
            "Require an exact match of the VersionNumber of all current values of the group's members. Require an "
            "exact match of the VersionNumber of all current or historized values of the group's members ")
        .value("historized", ctk::DataConsistencyGroup::MatchingMode::historized, "The data is not considered valid")
        .export_values();
  }

  /********************************************************************************************************************/

} // namespace ChimeraTK
