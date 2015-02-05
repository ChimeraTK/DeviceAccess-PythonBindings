/*
 * devBaseFactory.h
 *
 *  Created on: Feb 2, 2015
 *      Author: geo
 */

#ifndef DEVBASEFACTORY_H_
#define DEVBASEFACTORY_H_

#include "DeviceFactory.h"
#include <MtcaMappedDevice/devBase.h>

namespace mtca4upy {
/**
 * This is the concrete factory class responsible for creating a devBase object
 * wrapped in a devBaseAdapter. The factory returns either a devPCIE or a
 * DummyDevice object (wrapped within the devBase adapter). It uses the
 * information in _deviceDetails to decide which version to create.
 */
class devBaseFactory : public mtca4upy::DeviceFactory {
public:
  devBaseFactory(mtca4upy::DeviceInformation* deviceDetails);

  /*
   * Returns either an opened mtca4u::devPCIE device object or
   * mtca4u::DummyDevice object wrapped inside a devBase adapter (which is a
   * concrete implementation of PythonInterface). The PCIE version is returned,
   * if _deviceDetails->deviceName is the name of the PCIE device. The
   * DummyDevice version is  returned if the  _deviceDetails->deviceName is a
   * map file. The Map file is expected to be ASCII encoded and end in the
   * ".map" extension.
   */
  boost::shared_ptr<mtca4upy::PythonInterface> createDevice();
  ~devBaseFactory();

private:
  bool isDummyDevice();
  mtca4u::devBase* openDevice(mtca4u::devBase* device);
};

} /* namespace mtca4upy */

#endif /* DEVBASEFACTORY_H_ */
