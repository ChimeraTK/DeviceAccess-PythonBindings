// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <pybind11/pybind11.h>
// pybind11.h must come first
#include <ChimeraTK/VersionNumber.h>

#include <boost/date_time/posix_time/posix_time_types.hpp>

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/

  class PyVersionNumber : public VersionNumber {
   public:
    using VersionNumber::VersionNumber;

    PyVersionNumber() = default;
    PyVersionNumber(const PyVersionNumber&) = default;
    // NOLINTNEXTLINE(google-explicit-constructor)
    PyVersionNumber(const VersionNumber& other) : VersionNumber(other) {};
    ~PyVersionNumber() = default;

    std::string repr() const;

    static boost::posix_time::ptime getTime(ChimeraTK::VersionNumber& self);
    static PyVersionNumber getNullVersion();

    static void bind(py::module& mod);
  };

  /********************************************************************************************************************/

} // namespace ChimeraTK