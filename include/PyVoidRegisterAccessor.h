// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <pybind11/pybind11.h>
// pybind11.h must come first

#include "PyTransferElement.h"
#include "PyVersionNumber.h"

#include <ChimeraTK/AccessMode.h>
#include <ChimeraTK/SupportedUserTypes.h>
#include <ChimeraTK/VariantUserTypes.h>
#include <ChimeraTK/VoidRegisterAccessor.h>

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/

  class PyVoidRegisterAccessor : public VoidRegisterAccessor {
   public:
    using VoidRegisterAccessor::VoidRegisterAccessor;

    std::string repr(py::object& acc) const;

    static void bind(py::module& mod);

    bool write(const PyVersionNumber& versionNumber = PyVersionNumber{});
    bool writeDestructively(const PyVersionNumber& versionNumber = PyVersionNumber{});

    [[nodiscard]] py::list getAccessModeFlagsAsList() const { return accessModeFlagsToList(getAccessModeFlags()); }
  };

  /********************************************************************************************************************/

} // namespace ChimeraTK
