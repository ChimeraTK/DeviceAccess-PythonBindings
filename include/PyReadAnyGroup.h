// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <pybind11/pybind11.h>
// pybind11.h must come first

#include <ChimeraTK/ReadAnyGroup.h>

#include <pybind11/operators.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/

  class PyReadAnyGroup {
   public:
    static void bind(py::module& mod);
  };

  /********************************************************************************************************************/

  class PyReadAnyGroupNotification {
   public:
    static void bind(py::module& mod);
  };

  /********************************************************************************************************************/

} // namespace ChimeraTK
