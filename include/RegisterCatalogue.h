// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <ChimeraTK/RegisterCatalogue.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace DeviceAccessPython {

  /*****************************************************************************************************************/

  /**
   * Map RegisterCatalogue class to avoid dealing with RegisterPath objects in Python
   */
  class RegisterCatalogue {
   public:
    static py::list items(ChimeraTK::RegisterCatalogue& self);

    static py::list hiddenRegisters(ChimeraTK::RegisterCatalogue& self);

    static void bind(py::module& m);
  };

  /*****************************************************************************************************************/

  class RegisterInfo {
   public:
    // Translate return type from RegisterPath to string
    static std::string getRegisterName(ChimeraTK::RegisterInfo& self);

    static ChimeraTK::DataDescriptor getDataDescriptor(ChimeraTK::RegisterInfo& self);

    // convert return type form ChimeraTK::AccessModeFlags to Python list
    static py::list getSupportedAccessModes(ChimeraTK::RegisterInfo& self);

    static void bind(py::module& m);
    static void bindBackendRegisterInfoBase(py::module& m);
  };

  /*****************************************************************************************************************/

  class DataDescriptor {
   public:
    // Translate return type from RegisterPath to string
    static ChimeraTK::DataDescriptor::FundamentalType fundamentalType(ChimeraTK::DataDescriptor& self);

    static void bind(py::module& m);
  };

  /*****************************************************************************************************************/

} // namespace DeviceAccessPython
