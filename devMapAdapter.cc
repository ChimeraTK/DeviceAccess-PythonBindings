/*
 * devMapAdapter.cc
 *
 *  Created on: Jan 30, 2015
 *      Author: varghese
 */

#include "devMapAdapter.h"

namespace mtca4upy {

void devMapAdapter::readArea(int32_t regOffset, bp::numeric::array Buffer,
                             size_t size, uint8_t bar) {
  throwExceptionIfOutOfBounds(Buffer, size);
  _mappedDevice->readArea(regOffset, extractDataPointer(Buffer), size, bar);
}

void devMapAdapter::writeArea(uint32_t regOffset,
                              bp::numeric::array dataToWrite,
                              size_t bytesToWrite, uint8_t bar) {
  throwExceptionIfOutOfBounds(dataToWrite, bytesToWrite);
  _mappedDevice->writeArea(regOffset, extractDataPointer(dataToWrite),
                           bytesToWrite, bar);
}

void devMapAdapter::readDMA(uint32_t regOffset, bp::numeric::array Buffer,
                            size_t size, uint8_t bar) {
  throwExceptionIfOutOfBounds(Buffer, size);
  _mappedDevice->readDMA(regOffset, extractDataPointer(Buffer), size, bar);
}

void devMapAdapter::writeDMA(uint32_t regOffset, bp::numeric::array dataToWrite,
                             size_t bytesToWrite, uint8_t bar) {
  throwExceptionIfOutOfBounds(dataToWrite, bytesToWrite);
  _mappedDevice->writeDMA(regOffset, extractDataPointer(dataToWrite),
                          bytesToWrite, bar);
}

void devMapAdapter::readDMA(const std::string& regName,
                            bp::numeric::array bufferSpace, size_t dataSize,
                            uint32_t addRegOffset) {
  throwExceptionIfOutOfBounds(bufferSpace, dataSize);
  _mappedDevice->readDMA(regName, extractDataPointer(bufferSpace), dataSize,
                         addRegOffset);
}

void devMapAdapter::writeDMA(const std::string& regName,
                             bp::numeric::array dataToWrite, size_t dataSize,
                             uint32_t addRegOffset) {
  throwExceptionIfOutOfBounds(dataToWrite, dataSize);
  _mappedDevice->writeDMA(regName, extractDataPointer(dataToWrite), dataSize,
                          addRegOffset);
}

devMapAdapter::devMapAdapter(mtca4u::devMap<mtca4u::devBase>* mappedDevice)
    : _mappedDevice(mappedDevice) {}

devMapAdapter::~devMapAdapter() {
  // TODO Auto-generated destructor stub
}

} /* namespace mtcapy */

