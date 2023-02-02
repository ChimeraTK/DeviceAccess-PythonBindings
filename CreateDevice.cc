#include "PythonExceptions.h"
#include "PythonModuleMethods.h"

#include <ChimeraTK/DummyBackend.h>
#include <ChimeraTK/MapFileParser.h>

namespace mtca4upy {

  boost::shared_ptr<ChimeraTK::Device> createDevice(const std::string& deviceAlias) {
    ChimeraTK::Device* device = new ChimeraTK::Device();
    device->open(deviceAlias);
    return boost::shared_ptr<ChimeraTK::Device>(device);
  }

  boost::shared_ptr<ChimeraTK::Device> getDevice_no_alias() {
    ChimeraTK::Device* device = new ChimeraTK::Device();
    return boost::shared_ptr<ChimeraTK::Device>(device);
  }

  boost::shared_ptr<ChimeraTK::Device> getDevice(const std::string& deviceAlias) {
    ChimeraTK::Device* device = new ChimeraTK::Device(deviceAlias);
    return boost::shared_ptr<ChimeraTK::Device>(device);
  }

} /* namespace mtca4upy */
