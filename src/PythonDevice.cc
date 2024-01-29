// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "Device.h"
#include "HelperFunctions.h"

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

  ChimeraTK::VoidRegisterAccessor Device::getVoidRegisterAccessor(
      const ChimeraTK::Device& self, const std::string& registerPath, boost::python::list flaglist) {
    return self.getVoidRegisterAccessor(registerPath, convertFlagsFromPython(flaglist));
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

  ChimeraTK::AccessModeFlags Device::convertFlagsFromPython(boost::python::list flaglist) {
    ChimeraTK::AccessModeFlags flags{};
    size_t count = len((flaglist));
    for(size_t i = 0; i < count; i++) {
      flags.add(boost::python::extract<ChimeraTK::AccessMode>(flaglist.pop()));
    }
    return flags;
  }

  /*****************************************************************************************************************/

  void Device::activateAsyncRead(ChimeraTK::Device& self) {
    self.activateAsyncRead();
  }

  /*****************************************************************************************************************/

  ChimeraTK::RegisterCatalogue Device::getRegisterCatalogue(ChimeraTK::Device& self) {
    return self.getRegisterCatalogue();
  }

  /*****************************************************************************************************************/

  boost::python::numpy::ndarray Device::read(const ChimeraTK::Device& self, const std::string& registerPath,
      size_t numberOfElements, size_t elementsOffset, boost::python::list flaglist) {
    auto reg = self.getRegisterCatalogue().getRegister(registerPath);
    auto usertype = reg.getDataDescriptor().minimumDataType();

    std::unique_ptr<boost::python::numpy::ndarray> arr;

    ChimeraTK::callForTypeNoVoid(usertype, [&](auto arg) {
      using UserType = decltype(arg);
      auto acc = self.getTwoDRegisterAccessor<UserType>(
          registerPath, numberOfElements, elementsOffset, convertFlagsFromPython(flaglist));
      acc.read();
      arr = std::make_unique<boost::python::numpy::ndarray>(
          copyUserBufferToNpArray(acc, convert_usertype_to_dtype(usertype), reg.getNumberOfDimensions()));
    });

    return *arr;
  }

  /*****************************************************************************************************************/

  void Device::write(const ChimeraTK::Device& self, boost::python::numpy::ndarray& arr, const std::string& registerPath,
      size_t numberOfElements, size_t elementsOffset, boost::python::list flaglist) {
    auto usertype = convert_dytpe_to_usertype(arr.get_dtype());

    auto bufferTransfer = [&](auto arg) {
      auto acc = self.getTwoDRegisterAccessor<decltype(arg)>(
          registerPath, numberOfElements, elementsOffset, convertFlagsFromPython(flaglist));
      copyNpArrayToUserBuffer(acc, arr);
      acc.write();
    };

    ChimeraTK::callForTypeNoVoid(usertype, bufferTransfer);
  }

  /*****************************************************************************************************************/

} // namespace mtca4upy
