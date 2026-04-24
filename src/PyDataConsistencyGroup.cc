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
        .def(py::init<ctk::DataConsistencyGroup::MatchingMode>(), py::arg("matchingMode"),
            R"(Create a data consistency group.

            Args:
                matchingMode (MatchingMode): Matching strategy used to determine whether group members are consistent.)")
        .def(
            "add", [](ctk::DataConsistencyGroup& self, PyTransferElementBase& element) { self.add(element.getTE()); },
            py::arg("element"),
            R"(Add a TransferElement to the group.

            The same TransferElement can be part of multiple DataConsistencyGroup instances.

            Args:
                element (TransferElementBase): Element to add. It must be readable and use AccessMode.wait_for_new_data.)")
        .def("update", &ctk::DataConsistencyGroup::update, py::arg("updatedId"),
            R"(Process an update notification for one TransferElement.

            Call this after an update was received from ReadAnyGroup.

            Args:
                updatedId (TransferElementID): ID of the element that received an update.

            Returns:
                bool: True if the group is in a consistent state after processing the update. False if the updated ID
                was not added to this group.

            Note:
                For MatchingMode.historized, ReadAnyGroup only forwards consistent updates, so this function normally
                returns true.)")
        .def("getMatchingMode", &ctk::DataConsistencyGroup::getMatchingMode,
            R"(Get the current MatchingMode of this DataConsistencyGroup.

            Returns:
                MatchingMode: The matching mode used by this group.)")
        .def("isConsistent", &ctk::DataConsistencyGroup::isConsistent,
            R"(Check whether the group is currently in a consistent state.

            Returns:
                bool: `True` if a consistent state is reached, `False` otherwise.)");
  }

  /*******************************************************************************************************************/

  void PyMatchingMode::bind(py::module& m) {
    py::enum_<ctk::DataConsistencyGroup::MatchingMode>(
        m, "MatchingMode", "Enum describing the matching mode of a DataConsistencyGroup.")
        .value("none", ctk::DataConsistencyGroup::MatchingMode::none,
            "No consistency matching. Effectively disables consistency checks for the group.")
        .value("exact", ctk::DataConsistencyGroup::MatchingMode::exact,
            "Require an exact VersionNumber match across all current values of the group's members.")
        .value("historized", ctk::DataConsistencyGroup::MatchingMode::historized,
            "Allow matching against historized values to find a consistent state across group members.")
        .export_values();
  }

  /********************************************************************************************************************/

} // namespace ChimeraTK
