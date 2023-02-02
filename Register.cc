#include "PythonModuleMethods.h"

#include <ChimeraTK/RegisterInfo.h>

namespace mtca4upy::RegisterCatalogue {

  ChimeraTK::RegisterInfo getRegister(ChimeraTK::RegisterCatalogue& self, const std::string& registerPathName) {
    return self.getRegister(registerPathName);
  }

  bool hasRegister(ChimeraTK::RegisterCatalogue& self, const std::string& registerPathName) {
    return self.hasRegister(registerPathName);
  }

} /* namespace mtca4upy::RegisterCatalogue */

namespace mtca4upy::RegisterInfo {

  unsigned int getNumberOfElements(ChimeraTK::RegisterInfo& self) {
    return self.getNumberOfElements();
  }

  unsigned int getNumberOfChannels(ChimeraTK::RegisterInfo& self) {
    return self.getNumberOfChannels();
  }

  unsigned int getNumberOfDimensions(ChimeraTK::RegisterInfo& self) {
    return self.getNumberOfDimensions();
  }

  bool isReadable(ChimeraTK::RegisterInfo& self) {
    return self.isReadable();
  }

  bool isValid(ChimeraTK::RegisterInfo& self) {
    return self.isValid();
  }

  bool isWriteable(ChimeraTK::RegisterInfo& self) {
    return self.isWriteable();
  }

  std::string getRegisterName(ChimeraTK::RegisterInfo& self) {
    return self.getRegisterName();
  }

  boost::python::list getSupportedAccessModes(ChimeraTK::RegisterInfo& self) {
    ChimeraTK::AccessModeFlags flags = self.getSupportedAccessModes();
    boost::python::list python_flags{};
    if(flags.has(ChimeraTK::AccessMode::raw)) python_flags.append(ChimeraTK::AccessMode::raw);
    if(flags.has(ChimeraTK::AccessMode::wait_for_new_data))
      python_flags.append(ChimeraTK::AccessMode::wait_for_new_data);
    return python_flags;
  }
} /* namespace mtca4upy::RegisterInfo */