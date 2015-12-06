#include "PythonModuleMethods.h"
#include "PythonExceptions.h"

namespace mtca4upy {
namespace DeviceAccess {
  boost::shared_ptr<mtca4u::Device::RegisterAccessor> getRegisterAccessor(
      const mtca4u::Device& self, const std::string& moduleName,
      const std::string& regName) {
    return (self.getRegisterAccessor(regName, moduleName));
  }

  boost::shared_ptr<mtca4u::MultiplexedDataAccessor<float> >
  getMultiplexedDataAccessor(const mtca4u::Device& self,
                             const std::string& moduleName,
                             const std::string& regionName) {
    return (self.getCustomAccessor<mtca4u::MultiplexedDataAccessor<float> >(
        regionName, moduleName));
  }

  void writeRaw(mtca4u::Device& self, uint32_t regOffset,
                bp::numeric::array dataToWrite, size_t bytesToWrite,
                uint8_t bar) {
    throwExceptionIfOutOfBounds(dataToWrite, bytesToWrite);
    if (extractDataType(dataToWrite) == INT32) {
      int32_t* dataPointer =
          reinterpret_cast<int32_t*>(extractDataPointer(dataToWrite));
      self.writeArea(regOffset, dataPointer, bytesToWrite, bar);
      // self.writeArea(regOffset, dataPointer, bytesToWrite, bar);
    } else {
      throw mtca4upy::ArrayElementTypeNotSupported(
          "Data format used is unsupported");
    }
  }
} // namespace mtca4upy::deviceAccess
}
