#include "DeviceFactory.h"

namespace mtca4upy {

DeviceFactory::DeviceFactory(mtca4upy::DeviceInformation* deviceDetails)
    : _deviceDetails(deviceDetails) {}

DeviceFactory::~DeviceFactory() {
  // TODO Auto-generated destructor stub
}
DeviceInformation::DeviceInformation(std::string aliasName, std::string mapFile)
    : deviceName(aliasName), deviceMapFileLocation(mapFile) {}

std::string DeviceFactory::extractExtension(const std::string& fileName) {
  int fileNameLength =
      fileName.length(); // std::string.length returns number of bytes in the
                         // string. This should be the number of charachters,
                         // provided ASCII encoding is used for the file name
  std::string extension;
  if (fileNameLength >= 4) {
    extension = fileName.substr(fileNameLength - 4, 4);
  } else {
    extension = "";
  }
  return extension;
}
} /* namespace mtcapy */

