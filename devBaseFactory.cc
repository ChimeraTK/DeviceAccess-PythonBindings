#include "devBaseFactory.h"
#include "MtcaMappedDevice/DummyDevice.h"
#include "MtcaMappedDevice/devPCIE.h"
#include "devBaseAdapter.h"

namespace mtca4upy {

devBaseFactory::devBaseFactory(mtca4upy::DeviceInformation* deviceDetails)
    : mtca4upy::DeviceFactory(deviceDetails) {}

boost::shared_ptr<mtca4upy::PythonInterface> devBaseFactory::createDevice() {
  mtca4u::devBase* device;
  if (isDummyDevice() == true) {
    device = openDevice(new mtca4u::DummyDevice());
  } else {
    device = openDevice(new mtca4u::devPCIE());
  }
  return boost::shared_ptr<mtca4upy::PythonInterface>(
      new mtca4upy::devBaseAdapter(device));
}

devBaseFactory::~devBaseFactory() {
  // TODO Auto-generated destructor stub
}

bool devBaseFactory::isDummyDevice() {
  std::string deviceNameExtension =
      extractExtension(_deviceDetails->deviceName);
  return (deviceNameExtension == ".map");
}

mtca4u::devBase* devBaseFactory::openDevice(mtca4u::devBase* device) {
  device->openDev(_deviceDetails->deviceName);
  return device;
}

} /* namespace mtca4upy */

