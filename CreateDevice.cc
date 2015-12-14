#include "PythonModuleMethods.h"
#include <mtca4u/MapFileParser.h>
#include "PythonExceptions.h"

namespace mtca4upy {

/******************************************************************************/
// Forward function declarations. These are utility methods for createDevice.
static bool isDummyDevice(const std::string& deviceIdentifier,
                          const std::string& mapFile);
static boost::shared_ptr<mtca4u::Device> createMappedDevice(
    mtca4u::DeviceBackend* baseDevice, const std::string& mapFile);
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
  } else {
    return createMappedDevice(new mtca4u::PcieBackend(deviceIdentifier),
                              mapFile);
  }
}

static bool isDummyDevice(const std::string& deviceIdentifier,
                          const std::string& mapFile) {
return (deviceIdentifier == mapFile);
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

} /* namespace mtca4upy */
