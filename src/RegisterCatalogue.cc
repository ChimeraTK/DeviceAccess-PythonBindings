// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "RegisterCatalogue.h"

#include <pybind11/pybind11.h>
namespace ctk = ChimeraTK;

namespace py = pybind11;

namespace DeviceAccessPython {

  /*******************************************************************************************************************/

  py::list RegisterCatalogue::items(ChimeraTK::RegisterCatalogue& self) {
    py::list registerInfos{};
    for(const auto& regInfo : self) {
      registerInfos.append(ChimeraTK::RegisterInfo(regInfo.clone()));
    }
    return registerInfos;
  }

  /*******************************************************************************************************************/

  py::list RegisterCatalogue::hiddenRegisters(ChimeraTK::RegisterCatalogue& self) {
    py::list registerInfos{};
    for(const auto& regInfo : self.hiddenRegisters()) {
      registerInfos.append(ChimeraTK::RegisterInfo(regInfo.clone()));
    }
    return registerInfos;
  }

  /*******************************************************************************************************************/

  ChimeraTK::DataDescriptor RegisterInfo::getDataDescriptor(ChimeraTK::RegisterInfo& self) {
    return self.getDataDescriptor();
  }

  /*******************************************************************************************************************/

  std::string RegisterInfo::getRegisterName(ChimeraTK::RegisterInfo& self) {
    return self.getRegisterName();
  }

  /*******************************************************************************************************************/

  py::list RegisterInfo::getSupportedAccessModes(ChimeraTK::RegisterInfo& self) {
    ChimeraTK::AccessModeFlags flags = self.getSupportedAccessModes();
    py::list python_flags{};
    if(flags.has(ChimeraTK::AccessMode::raw)) python_flags.append(ChimeraTK::AccessMode::raw);
    if(flags.has(ChimeraTK::AccessMode::wait_for_new_data))
      python_flags.append(ChimeraTK::AccessMode::wait_for_new_data);
    return python_flags;
  }

  /*******************************************************************************************************************/

  ChimeraTK::DataDescriptor::FundamentalType DataDescriptor::fundamentalType(ChimeraTK::DataDescriptor& self) {
    return self.fundamentalType();
  }

  /*******************************************************************************************************************/

} /* namespace DeviceAccessPython*/
