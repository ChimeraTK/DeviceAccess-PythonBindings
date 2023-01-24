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

void mtca4upy::DeviceAccess::activateAsyncRead(ChimeraTK::Device& self) {
  self.activateAsyncRead();
}

ChimeraTK::RegisterCatalogue mtca4upy::DeviceAccess::getRegisterCatalogue(ChimeraTK::Device& self) {
  return self.getRegisterCatalogue();
}

namespace mtca4upy::VoidRegisterAccessor {

  bool write(ChimeraTK::VoidRegisterAccessor& self) { return self.write(); }

  bool writeDestructively(ChimeraTK::VoidRegisterAccessor& self) { return self.writeDestructively(); }

  void read(ChimeraTK::VoidRegisterAccessor& self) { return self.read(); }

  bool readNonBlocking(ChimeraTK::VoidRegisterAccessor& self) { return self.readNonBlocking(); }

  bool readLatest(ChimeraTK::VoidRegisterAccessor& self) { return self.readLatest(); }

} // namespace mtca4upy::VoidRegisterAccessor

namespace mtca4upy::ScalarRegisterAccessor {
  template<>
  void copyUserBufferToNpArray<ChimeraTK::Boolean>(
        ChimeraTK::ScalarRegisterAccessor<ChimeraTK::Boolean>& self, np::ndarray& np_buffer) {
      np_buffer[0] = static_cast<bool>(self);
    }
}

np::ndarray mtca4upy::DeviceAccess::read(
    const ChimeraTK::Device& self, np::ndarray& arr, const std::string& registerPath, boost::python::list flaglist) {
      uint8_t mul_data[][4] = {{1,2,3,4},{5,6,7,8},{1,3,5,7}};
      // np::dtype dt1 = np::dtype::get_builtin<uint8_t>();
      p::tuple shape = p::make_tuple(3,4);
      p::tuple stride = p::make_tuple(4,1);
      return np::from_data(mul_data, arr.get_dtype(), shape, stride,  p::object());
}
