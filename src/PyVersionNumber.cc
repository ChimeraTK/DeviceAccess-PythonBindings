// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyVersionNumber.h"

#include "string"

#include <ChimeraTK/VersionNumber.h>

#include <pybind11/stl.h>

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/

  std::string PyVersionNumber::repr() const {
    std::string rep{"<PyVersionNumber(versionNumber="};
    rep.append(std::string(static_cast<VersionNumber>(*this)));
    rep.append(")>");
    return rep;
  }
  boost::posix_time::ptime PyVersionNumber::getTime([[maybe_unused]] ChimeraTK::VersionNumber& self) {
    return boost::posix_time::ptime(boost::gregorian::date(1990, 1, 1));
  }

  /*****************************************************************************************************************/

  PyVersionNumber PyVersionNumber::getNullVersion() {
    return ChimeraTK::VersionNumber(nullptr);
  }

  /********************************************************************************************************************/

  void PyVersionNumber::bind(py::module& m) {
    py::class_<ChimeraTK::VersionNumber>(m, "VersionNumberBase")
        .def(
            "getVersionNumberAsString",
            [](const VersionNumber& self) -> std::string {
              std::ostringstream oss;
              oss << static_cast<ChimeraTK::VersionNumber>(self);
              return oss.str();
            },
            R"(Return the human readable string representation of the version number.

            Returns:
                str: Version number as string.)");

    py::class_<PyVersionNumber, ChimeraTK::VersionNumber>(m, "VersionNumber",
        R"(Class for generating and holding version numbers without exposing a numeric representation.

      Version numbers are used to resolve competing updates that are applied to the same process variable. For
      example, they can help in breaking an infinite update loop that might occur when two process variables are
      related and update each other.

      They are also used to determine the order of updates made to different process variables.)")
        .def(py::init<>())
        .def("getNullVersion", &PyVersionNumber::getNullVersion,
            R"(Get a VersionNumber which is not set (null version).

        The null version is guaranteed to be smaller than all version numbers generated with the default
        constructor and can be used to initialise version numbers that are not yet used for data transfers.

        Returns:
            VersionNumber: Null version number instance.)")
        .def("getTime", &PyVersionNumber::getTime,
            R"(Get the time stamp associated with this version number.

        Note:
          This Python binding currently returns a fixed placeholder time (1990-01-01 00:00:00).

        Returns:
            datetime: Time stamp of the version number.)")
        .def("__repr__", &PyVersionNumber::repr,
            R"(Return a debug representation including the version number string.

    Returns:
      str: Debug representation of the version number.)")
        .def(
            "__lt__",
            [](const ChimeraTK::VersionNumber& self, const ChimeraTK::VersionNumber& other) { return self < other; },
            py::arg("other"),
            R"(Compare whether this version number is smaller than another one.

    Args:
      other (VersionNumber): Version number to compare against.

    Returns:
      bool: True if self is smaller than other, otherwise False.)")
        .def(
            "__le__",
            [](const ChimeraTK::VersionNumber& self, const ChimeraTK::VersionNumber& other) { return self <= other; },
            py::arg("other"),
            R"(Compare whether this version number is smaller than or equal to another one.

    Args:
      other (VersionNumber): Version number to compare against.

    Returns:
      bool: True if self is smaller than or equal to other, otherwise False.)")
        .def(
            "__gt__",
            [](const ChimeraTK::VersionNumber& self, const ChimeraTK::VersionNumber& other) { return self > other; },
            py::arg("other"),
            R"(Compare whether this version number is greater than another one.

    Args:
      other (VersionNumber): Version number to compare against.

    Returns:
      bool: True if self is greater than other, otherwise False.)")
        .def(
            "__ge__",
            [](const ChimeraTK::VersionNumber& self, const ChimeraTK::VersionNumber& other) { return self >= other; },
            py::arg("other"),
            R"(Compare whether this version number is greater than or equal to another one.

    Args:
      other (VersionNumber): Version number to compare against.

    Returns:
      bool: True if self is greater than or equal to other, otherwise False.)")
        .def(
            "__ne__",
            [](const ChimeraTK::VersionNumber& self, const ChimeraTK::VersionNumber& other) { return self != other; },
            py::arg("other"),
            R"(Compare whether this version number is different from another one.

    Args:
      other (VersionNumber): Version number to compare against.

    Returns:
      bool: True if self differs from other, otherwise False.)")
        .def(
            "__eq__",
            [](const ChimeraTK::VersionNumber& self, const ChimeraTK::VersionNumber& other) { return self == other; },
            py::arg("other"),
            R"(Compare whether this version number is equal to another one.

    Args:
      other (VersionNumber): Version number to compare against.

    Returns:
      bool: True if self equals other, otherwise False.)");
  }

  /********************************************************************************************************************/

} // namespace ChimeraTK
