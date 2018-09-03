#include "PythonModuleMethods.h"
#include <mtca4u/MapFileParser.h>
#include <mtca4u/DummyBackend.h>
#include "PythonExceptions.h"

namespace mtca4upy {

boost::shared_ptr<mtca4u::Device> createDevice(
    const std::string& deviceAlias) {
  mtca4u::Device* device = new mtca4u::Device();
  device->open(deviceAlias);
  return boost::shared_ptr<mtca4u::Device>(device);
}

} /* namespace mtca4upy */
