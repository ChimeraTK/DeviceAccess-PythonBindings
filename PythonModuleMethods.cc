#include "PythonModuleMethods.h"

ChimeraTK::AccessModeFlags mtca4upy::DeviceAccess::convertFlagsFromPython(boost::python::list flaglist) {
  ChimeraTK::AccessModeFlags flags{};
  size_t count = len((flaglist));
  for(size_t i = 0; i < count; i++) {
    flags.add(p::extract<ChimeraTK::AccessMode>(flaglist.pop()));
  }
  return flags;
}

ChimeraTK::VoidRegisterAccessor mtca4upy::DeviceAccess::getVoidRegisterAccessor(
    const ChimeraTK::Device& self, const std::string& registerPath, boost::python::list flaglist) {
  return self.getVoidRegisterAccessor(registerPath, convertFlagsFromPython(flaglist));
}

namespace mtca4upy::VoidRegisterAccessor {

  bool write(ChimeraTK::VoidRegisterAccessor& self) { return self.write(); }

  bool writeDestructively(ChimeraTK::VoidRegisterAccessor& self) { return self.writeDestructively(); }

  void read(ChimeraTK::VoidRegisterAccessor& self) { return self.read(); }

  bool readNonBlocking(ChimeraTK::VoidRegisterAccessor& self) { return self.readNonBlocking(); }

  bool readLatest(ChimeraTK::VoidRegisterAccessor& self) { return self.readLatest(); }

} // namespace mtca4upy::VoidRegisterAccessor
