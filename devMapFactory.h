#ifndef MAPPEDDEVICEFACTORY_H_
#define MAPPEDDEVICEFACTORY_H_

#include "DeviceFactory.h"
#include <MtcaMappedDevice/devMap.h>
namespace mtca4upy {
/**
 * This is the concrete factory class responsible for creating a devMap object
 * wrapped in a devMapAdapter. The devMapAdapter implements the PythonInterface.
 * The factory returns either a Mapped version of devPCIE or a
 * DummyDevice object (wrapped within the devMapAdapter). It uses the
 * information in _deviceDetails to decide which version to create.
 */
class devMapFactory : public mtca4upy::DeviceFactory {
public:
  devMapFactory(mtca4upy::DeviceInformation* deviceDetails);
  /*
   * Returns either an opened mtca4u::devMap<mtca4u::devPCIE> device object or
   * mtca4u::devMap<mtca4u:DummyDevice> object wrapped inside a devMapAdapter
   * (which is a
   * concrete implementation of PythonInterface). The mapped PCIE version is
   * returned,
   * if _deviceDetails->deviceName is the name of the PCIE device and
   * _deviceDetails->deviceMapFileLocation is a valid register mapping file. The
   * DummyDevice version is  returned if both the  _deviceDetails->deviceName
   * and _deviceDetails->deviceMapFileLocation is the same valid register
   * mapping file.
   * The register mapping file is expected to be ASCII encoded and end with the
   * ".map" extension.
   */
  boost::shared_ptr<mtca4upy::PythonDevice> createDevice();
  ~devMapFactory();

private:
  mtca4u::devMap<mtca4u::devBase>* createMappedDevice(
      mtca4u::devBase* baseDevice);
  bool isDummyDevice();
  bool isPCIEDevice();
  void checkDummyDeviceParameters();
};

} /* namespace mtcapy */

#endif /* MAPPEDDEVICEFACTORY_H_ */
