// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <pybind11/pybind11.h>
// pybind11.h must come first

// #include "PyTransferElement.h"

#include <ChimeraTK/AccessMode.h>
#include <ChimeraTK/OneDRegisterAccessor.h>
#include <ChimeraTK/VariantUserTypes.h>

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/

  class PyOneDRegisterAccessor { // : public PyTransferElement<PyOneDRegisterAccessor> {
   public:
    template<typename T>
    using Vector = std::vector<T>;

    void set(const UserTypeTemplateVariantNoVoid<Vector>& vec);

    static void bind(py::module& mod);
  };

  /********************************************************************************************************************/

} // namespace ChimeraTK
