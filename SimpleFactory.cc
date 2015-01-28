/*
 * SimpleFactoty.cc
 *
 *  Created on: Jan 28, 2015
 *      Author: varghese
 */

#include "SimpleFactory.h"
#include <MtcaMappedDevice/devPCIE.h>
#include <MtcaMappedDevice/DummyDevice.h>

namespace mtca4upy {
Device::Device() {}

Device::~Device() {}

boost::shared_ptr<mtca4u::devBase> Device::createPCIEDevice() {
  return boost::shared_ptr<mtca4u::devBase>(new mtca4u::devPCIE);
}

boost::shared_ptr<mtca4u::devBase> Device::createDummyDevice() {
  return boost::shared_ptr<mtca4u::devBase>(new mtca4u::DummyDevice);
}
}
