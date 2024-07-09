// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "RegisterCatalogue.h"

namespace ctk = ChimeraTK;

namespace DeviceAccessPython {

  /*******************************************************************************************************************/

  ChimeraTK::RegisterInfo RegisterCatalogue::getRegister(
      ChimeraTK::RegisterCatalogue& self, const std::string& registerPathName) {
    return self.getRegister(registerPathName);
  }

  /*******************************************************************************************************************/

  bool RegisterCatalogue::hasRegister(ChimeraTK::RegisterCatalogue& self, const std::string& registerPathName) {
    return self.hasRegister(registerPathName);
  }

  /*******************************************************************************************************************/

    boost::python::list RegisterCatalogue::items(ChimeraTK::RegisterCatalogue& self) {
    boost::python::list registerInfos{};
    for(const auto &regInfo : self)
      registerInfos.append(ChimeraTK::RegisterInfo(regInfo.clone()));
    return registerInfos;
  }

  /*******************************************************************************************************************/

  std::string RegisterInfo::getRegisterName(ChimeraTK::RegisterInfo& self) {
    return self.getRegisterName();
  }

  /*******************************************************************************************************************/

  boost::python::list RegisterInfo::getSupportedAccessModes(ChimeraTK::RegisterInfo& self) {
    ChimeraTK::AccessModeFlags flags = self.getSupportedAccessModes();
    boost::python::list python_flags{};
    if(flags.has(ChimeraTK::AccessMode::raw)) python_flags.append(ChimeraTK::AccessMode::raw);
    if(flags.has(ChimeraTK::AccessMode::wait_for_new_data))
      python_flags.append(ChimeraTK::AccessMode::wait_for_new_data);
    return python_flags;
  }

  /*******************************************************************************************************************/

} /* namespace DeviceAccessPython*/