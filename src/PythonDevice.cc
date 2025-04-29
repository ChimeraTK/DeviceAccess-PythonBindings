// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "Device.h"
#include "HelperFunctions.h"

#include <ChimeraTK/AccessMode.h>
#include <ChimeraTK/Device.h>
#include <ChimeraTK/Utilities.h>

#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace DeviceAccessPython {

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
      const ChimeraTK::Device& self, const std::string& registerPath, const py::list& flaglist) {
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

  ChimeraTK::AccessModeFlags Device::convertFlagsFromPython(const py::list& flaglist) {
    ChimeraTK::AccessModeFlags flags{};
    for(auto flag : flaglist) {
      flags.add(flag.cast<ChimeraTK::AccessMode>());
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

  pybind11::array Device::read(const ChimeraTK::Device& self, const std::string& registerPath, size_t numberOfElements,
      size_t elementsOffset, const py::list& flaglist) {
    auto reg = self.getRegisterCatalogue().getRegister(registerPath);

    ChimeraTK::DataType usertype;
    if(!flaglist.contains(ChimeraTK::AccessMode::raw)) {
      usertype = reg.getDataDescriptor().minimumDataType();
    }
    else {
      usertype = reg.getDataDescriptor().rawDataType();
    }

    std::unique_ptr<pybind11::array> arr;

    ChimeraTK::callForTypeNoVoid(usertype, [&](auto arg) {
      using UserType = decltype(arg);
      auto acc = self.getTwoDRegisterAccessor<UserType>(
          registerPath, numberOfElements, elementsOffset, convertFlagsFromPython(flaglist));
      acc.read();
      arr = std::make_unique<pybind11::array>(
          copyUserBufferToNpArray(acc, ChimeraTK::convertUsertypeToDtype(usertype), reg.getNumberOfDimensions()));
    });

    return *arr;
  }

  /*****************************************************************************************************************/

  void Device::write(const ChimeraTK::Device& self, py::array& arr, const std::string& registerPath,
      size_t numberOfElements, size_t elementsOffset, const py::list& flaglist) {
    auto usertype = ChimeraTK::convertDTypeToUsertype(arr.dtype());

    auto bufferTransfer = [&](auto arg) {
      auto acc = self.getTwoDRegisterAccessor<decltype(arg)>(
          registerPath, numberOfElements, elementsOffset, convertFlagsFromPython(flaglist));
      copyNpArrayToUserBuffer(acc, arr);
      acc.write();
    };

    ChimeraTK::callForTypeNoVoid(usertype, bufferTransfer);
  }

  /*****************************************************************************************************************/

} // namespace DeviceAccessPython
