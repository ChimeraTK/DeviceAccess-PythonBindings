// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyVersionNumber.h"

#include "string"

#include <ChimeraTK/VersionNumber.h>

#include <pybind11/stl.h>

#include <iostream>

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
        .def("getVersionNumberAsString", [](const VersionNumber& self) -> std::string {
          std::ostringstream oss;
          oss << static_cast<ChimeraTK::VersionNumber>(self);
          return oss.str();
        });
    py::class_<PyVersionNumber, ChimeraTK::VersionNumber>(m, "VersionNumber",
        "Class for generating and holding version numbers without exposing a numeric representation.\n"
        "\n"
        "Version numbers are used to resolve competing updates that are applied to the same process variable. For "
        "example, it they can help in breaking an infinite update loop that might occur when two process variables "
        "are "
        "related and update each other.\n"
        "\n"
        "They are also used to determine the order of updates made to different process variables.\n"
        "\n")
        .def(py::init<>())
        .def("getNullVersion", &PyVersionNumber::getNullVersion,
            "Get a VersionNumber which is not set, i.e. the null version.")
        .def("getTime", &PyVersionNumber::getTime,
            "Get the time stamp associated with this version number.\n\n"
            "This is a dummy implementation which always returns 1990-01-01 00:00:00.")
        .def("__repr__", &PyVersionNumber::repr)
        .def("__lt__", &ChimeraTK::VersionNumber::operator<)
        .def("__le__", &ChimeraTK::VersionNumber::operator<=)
        .def("__gt__", &ChimeraTK::VersionNumber::operator>)
        .def("__ge__", &ChimeraTK::VersionNumber::operator>=)
        .def("__ne__", &ChimeraTK::VersionNumber::operator!=)
        .def("__eq__", &ChimeraTK::VersionNumber::operator==);
  }

  /********************************************************************************************************************/

} // namespace ChimeraTK
