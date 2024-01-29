// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PythonModuleMethods.h"

#include <ChimeraTK/SupportedUserTypes.h>

#include <limits>
#include <stdexcept>

namespace mtca4upy {

  /*****************************************************************************************************************/
  /* Implementations for class mtca4upy::Device */
  /*****************************************************************************************************************/

  ChimeraTK::AccessModeFlags Device::convertFlagsFromPython(boost::python::list flaglist) {
    ChimeraTK::AccessModeFlags flags{};
    size_t count = len((flaglist));
    for(size_t i = 0; i < count; i++) {
      flags.add(p::extract<ChimeraTK::AccessMode>(flaglist.pop()));
    }
    return flags;
  }

  /*****************************************************************************************************************/

  ChimeraTK::VoidRegisterAccessor Device::getVoidRegisterAccessor(
      const ChimeraTK::Device& self, const std::string& registerPath, boost::python::list flaglist) {
    return self.getVoidRegisterAccessor(registerPath, convertFlagsFromPython(flaglist));
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

  bool VoidRegisterAccessor::write(ChimeraTK::VoidRegisterAccessor& self) {
    return self.write();
  }

  /*****************************************************************************************************************/

  bool VoidRegisterAccessor::writeDestructively(ChimeraTK::VoidRegisterAccessor& self) {
    return self.writeDestructively();
  }

  /*****************************************************************************************************************/

  void VoidRegisterAccessor::read(ChimeraTK::VoidRegisterAccessor& self) {
    return self.read();
  }

  /*****************************************************************************************************************/

  bool VoidRegisterAccessor::readNonBlocking(ChimeraTK::VoidRegisterAccessor& self) {
    return self.readNonBlocking();
  }

  /*****************************************************************************************************************/

  bool VoidRegisterAccessor::readLatest(ChimeraTK::VoidRegisterAccessor& self) {
    return self.readLatest();
  }

  /*****************************************************************************************************************/

  np::ndarray Device::read(const ChimeraTK::Device& self, const std::string& registerPath, size_t numberOfElements,
      size_t elementsOffset, boost::python::list flaglist) {
    auto reg = self.getRegisterCatalogue().getRegister(registerPath);
    auto usertype = reg.getDataDescriptor().minimumDataType();

    std::unique_ptr<np::ndarray> arr;

    ChimeraTK::callForTypeNoVoid(usertype, [&](auto arg) {
      using UserType = decltype(arg);
      auto acc = self.getTwoDRegisterAccessor<UserType>(
          registerPath, numberOfElements, elementsOffset, convertFlagsFromPython(flaglist));
      acc.read();
      arr = std::make_unique<np::ndarray>(
          copyUserBufferToNpArray(acc, convert_usertype_to_dtype(usertype), reg.getNumberOfDimensions()));
    });

    return *arr;
  }

  /*****************************************************************************************************************/

  void Device::write(const ChimeraTK::Device& self, np::ndarray& arr, const std::string& registerPath,
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
  /** Implementations for free functions in namespace mtca4upy */
  /*****************************************************************************************************************/

  ChimeraTK::DataType convert_dytpe_to_usertype(np::dtype dtype) {
    if(dtype == np::dtype::get_builtin<int8_t>()) {
      return ChimeraTK::DataType::int8;
    }
    if(dtype == np::dtype::get_builtin<int16_t>()) {
      return ChimeraTK::DataType::int16;
    }
    if(dtype == np::dtype::get_builtin<int32_t>()) {
      return ChimeraTK::DataType::int32;
    }
    if(dtype == np::dtype::get_builtin<int64_t>()) {
      return ChimeraTK::DataType::int64;
    }
    if(dtype == np::dtype::get_builtin<uint8_t>()) {
      return ChimeraTK::DataType::uint8;
    }
    if(dtype == np::dtype::get_builtin<uint16_t>()) {
      return ChimeraTK::DataType::uint16;
    }
    if(dtype == np::dtype::get_builtin<uint32_t>()) {
      return ChimeraTK::DataType::uint32;
    }
    if(dtype == np::dtype::get_builtin<uint64_t>()) {
      return ChimeraTK::DataType::uint64;
    }
    if(dtype == np::dtype::get_builtin<float>()) {
      return ChimeraTK::DataType::float32;
    }
    if(dtype == np::dtype::get_builtin<double>()) {
      return ChimeraTK::DataType::float64;
    }
    if(dtype == np::dtype::get_builtin<bool>()) {
      return ChimeraTK::DataType::Boolean;
    }
    throw std::invalid_argument("Unsupported numpy dtype");
  }

  /*****************************************************************************************************************/

  np::dtype convert_usertype_to_dtype(ChimeraTK::DataType usertype) {
    std::unique_ptr<np::dtype> rv;
    ChimeraTK::callForTypeNoVoid(usertype, [&](auto arg) {
      using UserType = decltype(arg);
      if constexpr(std::is_same<UserType, ChimeraTK::Boolean>::value) {
        rv = std::make_unique<np::dtype>(np::dtype::get_builtin<bool>());
      }
      if constexpr(std::is_same<UserType, std::string>::value) {
        rv = std::make_unique<np::dtype>(p::make_tuple('U', 1));
      }
      else {
        rv = std::make_unique<np::dtype>(np::dtype::get_builtin<UserType>());
      }
    });
    return *rv;
  }

  /*****************************************************************************************************************/

  std::string convertStringFromPython(size_t linearIndex, np::ndarray& np_buffer) {
    // Note: it is unclear why the conversion in this direction has to be so complicated, while in the other
    // direction an assignment to an std::string is sufficient.

    // create C++ 4-byte string of matching length
    size_t itemsize = np_buffer.get_dtype().get_itemsize();
    assert(itemsize % sizeof(char32_t) == 0);
    std::u32string widestring;
    widestring.resize(itemsize / 4);

    // copy string to C++ buffer
    memcpy(widestring.data(), np_buffer.get_data() + itemsize * linearIndex, itemsize);

    // convert to UTF-8 string and store to accessor
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
    return conv.to_bytes(widestring);
  }

  /*****************************************************************************************************************/

} // namespace mtca4upy
