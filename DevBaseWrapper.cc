/*
 * DevBaseWrapper.cc
 *
 *  Created on: Jan 28, 2015
 *      Author: varghese
 */

#include "DevBaseWrapper.h"

namespace mtca4upy { // TODO: Refactor to a better name
DevBaseWrapper::DevBaseWrapper() {
  // TODO Auto-generated constructor stub
}

DevBaseWrapper::~DevBaseWrapper() {
  // TODO Auto-generated destructor stub
}

void DevBaseWrapper::openDev(const std::string& devName, int perm,
                             mtca4u::devConfigBase* pConfig) {}

void DevBaseWrapper::closeDev() {}

void DevBaseWrapper::readReg(uint32_t regOffset, int32_t* data, uint8_t bar) {}

void DevBaseWrapper::writeReg(uint32_t regOffset, int32_t data, uint8_t bar) {}

void DevBaseWrapper::readArea(uint32_t regOffset, int32_t* data, size_t size,
                              uint8_t bar) {}

void DevBaseWrapper::writeArea(uint32_t regOffset, const int32_t* data,
                               size_t size, uint8_t bar) {}

void DevBaseWrapper::readDMA(uint32_t regOffset, int32_t* data, size_t size,
                             uint8_t bar) {}

void DevBaseWrapper::writeDMA(uint32_t regOffset, const int32_t* data,
                              size_t size, uint8_t bar) {}

void DevBaseWrapper::readDeviceInfo(std::string* devInfo) {}

bool DevBaseWrapper::isOpen() { return false; }
}
