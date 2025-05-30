// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PythonModuleMethods.h"

#include <ChimeraTK/SupportedUserTypes.h>

#include <limits>
#include <stdexcept>

namespace DeviceAccessPython {

  /********************************************************************************************************************/

  boost::shared_ptr<ChimeraTK::Device> createDevice(const std::string& deviceAlias) {
    ChimeraTK::Device* device = new ChimeraTK::Device();
    device->open(deviceAlias);
    return boost::shared_ptr<ChimeraTK::Device>(device);
  }

  /********************************************************************************************************************/

  boost::shared_ptr<ChimeraTK::Device> getDevice_no_alias() {
    ChimeraTK::Device* device = new ChimeraTK::Device();
    return boost::shared_ptr<ChimeraTK::Device>(device);
  }

  /********************************************************************************************************************/

  boost::shared_ptr<ChimeraTK::Device> getDevice(const std::string& deviceAlias) {
    ChimeraTK::Device* device = new ChimeraTK::Device(deviceAlias);
    return boost::shared_ptr<ChimeraTK::Device>(device);
  }

  /********************************************************************************************************************/

  void setDmapFile(const std::string& dmapFile) {
    ChimeraTK::setDMapFilePath(dmapFile);
  }

  /********************************************************************************************************************/

  std::string getDmapFile() {
    return (ChimeraTK::getDMapFilePath());
  }
  /********************************************************************************************************************/

} // namespace DeviceAccessPython
