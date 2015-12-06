#include "PythonModuleMethods.h"
#include <mtca4u/MapFileParser.h>
#include "PythonExceptions.h"

namespace mtca4upy {

/******************************************************************************/
// Forward function declarations. These are utility methods for createDevice.
static bool isDummyDevice(const std::string& deviceIdentifier,
                          const std::string& mapFile);
static const std::string extractExtension(const std::string& fileName);
static boost::shared_ptr<mtca4u::Device> createMappedDevice(
    mtca4u::DeviceBackend* baseDevice, const std::string& mapFile);
static bool isPCIEDevice(const std::string& deviceIdentifier,
                         const std::string& mapFile);
/******************************************************************************/

boost::shared_ptr<mtca4u::Device> createDevice(
    const std::string& deviceAlias) {
  mtca4u::Device* device = new mtca4u::Device();
  device->open(deviceAlias);
  return boost::shared_ptr<mtca4u::Device>(device);
}

boost::shared_ptr<mtca4u::Device> createDevice(
    const std::string& deviceIdentifier, const std::string& mapFile) {
  if (isDummyDevice(deviceIdentifier, mapFile) == true) {
    return createMappedDevice(new mtca4u::DummyBackend(deviceIdentifier),
                              mapFile);
  } else if (isPCIEDevice(deviceIdentifier, mapFile) == true) {
    return createMappedDevice(new mtca4u::PcieBackend(deviceIdentifier),
                              mapFile);
  } else {
    throw mtca4upy::DeviceNotSupported("Unable to identify device");
  }
}

static bool isDummyDevice(const std::string& deviceIdentifier,
                          const std::string& mapFile) {
return (deviceIdentifier == mapFile);
}

static const std::string extractExtension(const std::string& fileName) {
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

static boost::shared_ptr<mtca4u::Device> createMappedDevice(
    mtca4u::DeviceBackend* baseDevice, const std::string& mapFile) {

  boost::shared_ptr<mtca4u::DeviceBackend> device(baseDevice);
  boost::shared_ptr<mtca4u::Device> mappedDevice(new mtca4u::Device());
  boost::shared_ptr<mtca4u::RegisterInfoMap> ptrmapFile =
      mtca4u::MapFileParser().parse(mapFile);
  mappedDevice->open(device, ptrmapFile);
  return mappedDevice;
}

static bool isPCIEDevice(const std::string& deviceIdentifier,
                         const std::string& mapFile) {
  std::string deviceNameExtension = extractExtension(deviceIdentifier);
  std::string mapFileExtension = extractExtension(mapFile);
  return ((deviceNameExtension != ".map") && (mapFileExtension == ".map"));
}
} /* namespace mtca4upy */
