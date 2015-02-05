/*
 * devBaseAdapter.cc
 *
 *  Created on: Feb 1, 2015
 *      Author: geo
 */

#include "devBaseAdapter.h"

namespace mtca4upy {

devBaseAdapter::devBaseAdapter(mtca4u::devBase* mtcaDevice)
    : _mtcaDevice(mtcaDevice) {}

void devBaseAdapter::readArea(int32_t regOffset, bp::numeric::array Buffer,
                              size_t size, uint8_t bar) {
  _mtcaDevice->readArea(regOffset, extractDataPointer(Buffer), size, bar);
}

void devBaseAdapter::writeArea(uint32_t regOffset,
                               bp::numeric::array dataToWrite,
                               size_t bytesToWrite, uint8_t bar) {
  throwExceptionIfOutOfBounds(dataToWrite, bytesToWrite);
  _mtcaDevice->writeArea(regOffset, extractDataPointer(dataToWrite),
                         bytesToWrite, bar);
}

void devBaseAdapter::readDMA(uint32_t regOffset, bp::numeric::array Buffer,
                             size_t size, uint8_t bar) {
  _mtcaDevice->readDMA(regOffset, extractDataPointer(Buffer), size, bar);
}

void devBaseAdapter::writeDMA(uint32_t regOffset,
                              bp::numeric::array dataToWrite, size_t size,
                              uint8_t bar) {
  throwExceptionIfOutOfBounds(dataToWrite, size);
  _mtcaDevice->writeDMA(regOffset, extractDataPointer(dataToWrite), size, bar);
}

void devBaseAdapter::readDMA(const std::string& regName,
                             bp::numeric::array bufferSpace, size_t dataSize,
                             uint32_t addRegOffset) {
  throw mtca4upy::MethodNotImplementedException();
}

void devBaseAdapter::writeDMA(const std::string& regName,
                              bp::numeric::array dataToWrite, size_t dataSize,
                              uint32_t addRegOffset) {
  throw MethodNotImplementedException();
}

devBaseAdapter::~devBaseAdapter() {
  // TODO Auto-generated destructor stub
}

} /* namespace mtcapy */
