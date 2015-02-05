/*
 * PythonObjectFactory.cc
 *
 *  Created on: Feb 2, 2015
 *      Author: geo
 */

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
  std::string extension = fileName.substr(fileName.length() - 4, 4);
  return extension;
}
} /* namespace mtcapy */

