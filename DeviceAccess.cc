#include "PythonExceptions.h"
#include "PythonModuleMethods.h"
#include <mtca4u/Device.h>
#include <mtca4u/Utilities.h>

namespace mtca4upy {
namespace DeviceAccess {

  mtca4u::TwoDRegisterAccessor<float> getTwoDAccessor(
      const mtca4u::Device& self, const std::string& registerPath) {
    return (self.getTwoDRegisterAccessor<float>(registerPath));
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

  mtca4u::OneDRegisterAccessor<int32_t> getRawOneDAccessor(
      const mtca4u::Device& self, const std::string& registerPath,
      size_t numberOfelementsToRead, size_t elementOffset) {
    return self.getOneDRegisterAccessor<int32_t>(
        registerPath, numberOfelementsToRead, elementOffset,
        { mtca4u::AccessMode::raw });
  }

} // namespace mtca4upy::deviceAccess
}
void mtca4upy::setDmapFile(const std::string& dmapFile) {
  mtca4u::setDMapFilePath(dmapFile);
}

std::string mtca4upy::getDmapFile() {
  return (mtca4u::getDMapFilePath());
}
