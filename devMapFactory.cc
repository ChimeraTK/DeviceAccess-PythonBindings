#include <MtcaMappedDevice/DummyDevice.h>
#include <MtcaMappedDevice/devPCIE.h>
#include "devMapAdapter.h"
#include <MtcaMappedDevice/mapFileParser.h>
#include "devMapFactory.h"
#include "PythonExceptions.h"

namespace mtca4upy {

devMapFactory::devMapFactory(mtca4upy::DeviceInformation* deviceDetails)
    : mtca4upy::DeviceFactory(deviceDetails) {}

boost::shared_ptr<mtca4upy::PythonInterface> devMapFactory::createDevice() {
  mtca4u::devMap<mtca4u::devBase>* mappedDevice;
  if (isDummyDevice() == true) {
    checkDummyDeviceParameters();
    mappedDevice = createMappedDevice(new mtca4u::DummyDevice());
  } else {
    mappedDevice = createMappedDevice(new mtca4u::devPCIE());
  }
  return boost::shared_ptr<mtca4upy::PythonInterface>(
      new mtca4upy::devMapAdapter(mappedDevice));
}

mtca4u::devMap<mtca4u::devBase>* devMapFactory::createMappedDevice(
    mtca4u::devBase* baseDevice) {

  boost::shared_ptr<mtca4u::devBase> device(baseDevice);
  device->openDev(_deviceDetails->deviceName);

  mtca4u::devMap<mtca4u::devBase>* mappedDevice =
      new mtca4u::devMap<mtca4u::devBase>();

  boost::shared_ptr<mtca4u::mapFile> ptrmapFile =
      mtca4u::mapFileParser().parse(_deviceDetails->deviceMapFileLocation);

  mappedDevice->openDev(device, ptrmapFile);
  return mappedDevice;
}

devMapFactory::~devMapFactory() {
  // TODO Auto-generated destructor stub
}

bool devMapFactory::isDummyDevice() {
  std::string deviceNameExtension =
      extractExtension(_deviceDetails->deviceName);
  std::string mapFileExtension =
      extractExtension(_deviceDetails->deviceMapFileLocation);
  return ((deviceNameExtension == ".map") && (mapFileExtension == ".map"));
}

void devMapFactory::checkDummyDeviceParameters() {
  if (_deviceDetails->deviceName != _deviceDetails->deviceMapFileLocation) {
    throw DummyDeviceBadParameterException();
  }
}

} /* namespace mtcapy */

