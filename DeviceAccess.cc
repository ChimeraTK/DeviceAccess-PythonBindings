#include "PythonExceptions.h"
#include "PythonModuleMethods.h"
#include <ChimeraTK/Device.h>
#include <ChimeraTK/Utilities.h>

namespace mtca4upy { namespace DeviceAccess {

  ChimeraTK::TwoDRegisterAccessor<double> getTwoDAccessor(
      const ChimeraTK::Device& self, const std::string& registerPath) {
    return (self.getTwoDRegisterAccessor<double>(registerPath));
  }

  void writeRaw(ChimeraTK::Device& self, std::string const& registerName, //
      uint32_t regOffset, mtca4upy::NumpyObject dataToWrite, size_t bytesToWrite) {
    throwExceptionIfOutOfBounds(dataToWrite, bytesToWrite);

    if(extractDataType(dataToWrite) == INT32) {
      int32_t* dataPointer = reinterpret_cast<int32_t*>(extractDataPointer(dataToWrite));

      int wordsToWrite = bytesToWrite / 4;
      std::vector<int32_t> data(dataPointer, dataPointer + wordsToWrite);
      self.write(registerName, data, regOffset, {ChimeraTK::AccessMode::raw});
    }
    else {
      throw mtca4upy::ArrayElementTypeNotSupported("Data format used is unsupported");
    }
  }

  ChimeraTK::OneDRegisterAccessor<int32_t> getRawOneDAccessor(const ChimeraTK::Device& self,
      const std::string& registerPath, size_t numberOfelementsToRead, size_t elementOffset) {
    return self.getOneDRegisterAccessor<int32_t>(
        registerPath, numberOfelementsToRead, elementOffset, {ChimeraTK::AccessMode::raw});
  }

  std::string getCatalogueMetadata(const ChimeraTK::Device& self, const std::string& parameterName) {
    //return self.getMetadataCatalogue().getMetadata(parameterName);
    return "temporary fix";
  }

  void open(ChimeraTK::Device& self) { self.open(); }
  void open(ChimeraTK::Device& self, std::string const& aliasName) { self.open(aliasName); }
  void close(ChimeraTK::Device& self) { self.close(); }

}} // namespace mtca4upy::DeviceAccess
void mtca4upy::setDmapFile(const std::string& dmapFile) {
  ChimeraTK::setDMapFilePath(dmapFile);
}

std::string mtca4upy::getDmapFile() {
  return (ChimeraTK::getDMapFilePath());
}
