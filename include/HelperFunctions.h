// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "PyVersionNumber.h"

#include <ChimeraTK/NDRegisterAccessorAbstractor.h>
#include <ChimeraTK/SupportedUserTypes.h>
#include <ChimeraTK/VariantUserTypes.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ChimeraTK {

  /*****************************************************************************************************************/

  ChimeraTK::DataType convertDTypeToUsertype(const py::dtype& dtype);

  py::dtype convertUsertypeToDtype(const ChimeraTK::DataType& usertype);

  PyVersionNumber getNewVersionNumberIfNull(const PyVersionNumber& versionNumber);

  py::array convertPyListToNumpyArray(const py::list& list, const py::dtype& dtype);

  [[nodiscard]] py::list accessModeFlagsToList(const ChimeraTK::AccessModeFlags& flags);

  /*****************************************************************************************************************/

} // namespace ChimeraTK
