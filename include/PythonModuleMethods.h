// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <ChimeraTK/Device.h>

/*****************************************************************************************************************/

namespace DeviceAccessPython {

  /*
   * This method uses the factory provided by the device access library for device
   * creation. The deviceAlias is picked from the specified dmap file, which is
   * set through the environment variable DMAP_PATH_ENV
   */

  boost::shared_ptr<ChimeraTK::Device> createDevice(const std::string& deviceAlias);
  boost::shared_ptr<ChimeraTK::Device> getDevice_no_alias();
  boost::shared_ptr<ChimeraTK::Device> getDevice(const std::string& deviceAlias);

  void setDmapFile(const std::string& dmapFile);
  std::string getDmapFile();

  /*****************************************************************************************************************/

} // namespace DeviceAccessPython
