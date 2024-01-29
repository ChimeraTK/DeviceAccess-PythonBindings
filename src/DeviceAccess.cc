// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PythonExceptions.h"
#include "PythonModuleMethods.h"

#include <ChimeraTK/Device.h>
#include <ChimeraTK/Utilities.h>

namespace mtca4upy {

  /*****************************************************************************************************************/

  ChimeraTK::TwoDRegisterAccessor<double> Device::getTwoDAccessor(
      const ChimeraTK::Device& self, const std::string& registerPath) {
    return (self.getTwoDRegisterAccessor<double>(registerPath));
  }

  /*****************************************************************************************************************/

  ChimeraTK::OneDRegisterAccessor<int32_t> Device::getRawOneDAccessor(const ChimeraTK::Device& self,
      const std::string& registerPath, size_t numberOfelementsToRead, size_t elementOffset) {
    return self.getOneDRegisterAccessor<int32_t>(
        registerPath, numberOfelementsToRead, elementOffset, {ChimeraTK::AccessMode::raw});
  }

  /*****************************************************************************************************************/

  std::string Device::getCatalogueMetadata(const ChimeraTK::Device& self, const std::string& parameterName) {
    return self.getMetadataCatalogue().getMetadata(parameterName);
  }

  /*****************************************************************************************************************/

  void Device::open(ChimeraTK::Device& self) {
    self.open();
  }

  /*****************************************************************************************************************/

  void Device::open(ChimeraTK::Device& self, std::string const& aliasName) {
    self.open(aliasName);
  }

  /*****************************************************************************************************************/

  void Device::close(ChimeraTK::Device& self) {
    self.close();
  }

  /*****************************************************************************************************************/
  /*****************************************************************************************************************/

  boost::posix_time::ptime VersionNumber::getTime([[maybe_unused]] ChimeraTK::VersionNumber& self) {
    return boost::posix_time::ptime(boost::gregorian::date(1990, 1, 1));
  }

  /*****************************************************************************************************************/

  ChimeraTK::VersionNumber VersionNumber::getNullVersion() {
    return ChimeraTK::VersionNumber(nullptr);
  }

  /*****************************************************************************************************************/
  /*****************************************************************************************************************/

  void setDmapFile(const std::string& dmapFile) {
    ChimeraTK::setDMapFilePath(dmapFile);
  }

  /*****************************************************************************************************************/

  std::string getDmapFile() {
    return (ChimeraTK::getDMapFilePath());
  }

  /*****************************************************************************************************************/

} // namespace mtca4upy
