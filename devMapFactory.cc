#include <MtcaMappedDevice/DummyDevice.h>
#include <MtcaMappedDevice/devPCIE.h>
#include "devMapAdapter.h"
#include <MtcaMappedDevice/mapFileParser.h>
#include "devMapFactory.h"
#include "PythonExceptions.h"

namespace mtca4upy {

devMapFactory::devMapFactory(mtca4upy::DeviceInformation* deviceDetails)
    : mtca4upy::DeviceFactory(deviceDetails) {}

boost::shared_ptr<mtca4upy::PythonDevice> devMapFactory::createDevice() {
  mtca4u::devMap<mtca4u::devBase>* mappedDevice;
  // TODO: Refactor the checks for the device. Maybe later introduce a return
  // type as an enum which indicate the type of method (instead of the
  // individual checks?)
  if (isDummyDevice() == true) {
    checkDummyDeviceParameters();
    mappedDevice = createMappedDevice(new mtca4u::DummyDevice());
  } else if (isPCIEDevice() == true) {
    mappedDevice = createMappedDevice(new mtca4u::devPCIE());
  } else {
    throw mtca4upy::DeviceNotSupported("Unable to identify device");
  }
  return boost::shared_ptr<mtca4upy::PythonDevice>(
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
    throw DummyDeviceBadParameterException("Mapped Dummy Device expects first "
                                           "and second parameters to be the "
                                           "same map file");
  }
}

} /* namespace mtcapy */

bool mtca4upy::devMapFactory::isPCIEDevice() {
  std::string deviceNameExtension =
      extractExtension(_deviceDetails->deviceName);
  std::string mapFileExtension =
      extractExtension(_deviceDetails->deviceMapFileLocation);
  return ((deviceNameExtension != ".map") && (mapFileExtension == ".map"));
}
