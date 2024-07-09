// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <ChimeraTK/RegisterCatalogue.h>

#include <boost/python/list.hpp>

namespace DeviceAccessPython {

  /*****************************************************************************************************************/

  /**
   * Map RegisterCatalogue class to avoid dealing with RegisterPath objects in Python
   */
  class RegisterCatalogue {
   public:
    static bool hasRegister(ChimeraTK::RegisterCatalogue& self, const std::string& registerPathName);
    static ChimeraTK::RegisterInfo getRegister(ChimeraTK::RegisterCatalogue& self, const std::string& registerPathName);
    static boost::python::list items(ChimeraTK::RegisterCatalogue& self);
  };

  /*****************************************************************************************************************/

  class RegisterInfo {
   public:
    // Translate return type from RegisterPath to string
    static std::string getRegisterName(ChimeraTK::RegisterInfo& self);

    // convert return type form ChimeraTK::AccessModeFlags to Python list
    static boost::python::list getSupportedAccessModes(ChimeraTK::RegisterInfo& self);
  };

  /*****************************************************************************************************************/

} // namespace DeviceAccessPython