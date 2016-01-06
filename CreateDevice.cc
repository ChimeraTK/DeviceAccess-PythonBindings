#include "PythonModuleMethods.h"
#include <mtca4u/MapFileParser.h>
#include "PythonExceptions.h"

namespace mtca4upy {

/******************************************************************************/
// Forward function declarations. These are utility methods for createDevice.
static bool isDummyDevice(const std::string& deviceIdentifier,
                          const std::string& mapFile);
static boost::shared_ptr<mtca4u::Device> createDevice(
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
    return createDevice(new mtca4u::DummyBackend(deviceIdentifier),
                              mapFile);
  } else {
    return createDevice(new mtca4u::PcieBackend(deviceIdentifier),
                              mapFile);
  }
}

static bool isDummyDevice(const std::string& deviceIdentifier,
                          const std::string& mapFile) {
return (deviceIdentifier == mapFile);
}

static boost::shared_ptr<mtca4u::Device> createDevice(
    mtca4u::DeviceBackend* backendRawPtr, const std::string& mapFile) {

  boost::shared_ptr<mtca4u::DeviceBackend> backend(backendRawPtr);
  boost::shared_ptr<mtca4u::Device> device(new mtca4u::Device());
  boost::shared_ptr<mtca4u::RegisterInfoMap> registerMap =
      mtca4u::MapFileParser().parse(mapFile);
  device->open(backend, registerMap);
  return device;
}

} /* namespace mtca4upy */
