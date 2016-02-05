#include "PythonModuleMethods.h"
#include <mtca4u/MapFileParser.h>
#include <mtca4u/DummyBackend.h>
#include "PythonExceptions.h"

namespace mtca4upy {

/******************************************************************************/
// Forward function declarations. These are utility methods for createDevice.
static bool isDummyDevice(const std::string& deviceIdentifier,
                          const std::string& mapFile);
static boost::shared_ptr<mtca4u::Device> createDevice(
    mtca4u::DeviceBackend* baseDevice);
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
    return createDevice(new mtca4u::DummyBackend(deviceIdentifier));
  } else {
    return createDevice(new mtca4u::PcieBackend(deviceIdentifier, mapFile));
  }
}

static bool isDummyDevice(const std::string& deviceIdentifier,
                          const std::string& mapFile) {
return (deviceIdentifier == mapFile);
}

static boost::shared_ptr<mtca4u::Device> createDevice(
    mtca4u::DeviceBackend* backendRawPtr) {

  boost::shared_ptr<mtca4u::DeviceBackend> backend(backendRawPtr);
  boost::shared_ptr<mtca4u::Device> device(new mtca4u::Device());
  device->open(backend);
  return device;
}

} /* namespace mtca4upy */
