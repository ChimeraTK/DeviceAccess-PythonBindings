#include "devMapAdapter.h"

namespace mtca4upy {

void devMapAdapter::readRaw(uint32_t regOffset, bp::numeric::array Buffer,
                            size_t size, uint8_t bar) {
  throwExceptionIfOutOfBounds(Buffer, size);
  _mappedDevice->readArea(regOffset, mtca4upy::extractDataPointer(Buffer), size, bar);
}

void devMapAdapter::writeRaw(uint32_t regOffset, bp::numeric::array dataToWrite,
                             size_t bytesToWrite, uint8_t bar) {
  throwExceptionIfOutOfBounds(dataToWrite, bytesToWrite);
  _mappedDevice->writeArea(regOffset, mtca4upy::extractDataPointer(dataToWrite),
                           bytesToWrite, bar);
}

void devMapAdapter::readDMA(uint32_t regOffset, bp::numeric::array Buffer,
                            size_t size) {
  uint8_t dummyDMABar = 0; // The value should not matter; MappedDevice readDMA
                           // would not be dependent on the pcie bar
  throwExceptionIfOutOfBounds(Buffer, size);
  _mappedDevice->readDMA(regOffset, mtca4upy::extractDataPointer(Buffer), size,
                         dummyDMABar);
}

void devMapAdapter::writeDMA(uint32_t regOffset, bp::numeric::array dataToWrite,
                             size_t bytesToWrite) {
  uint8_t dummyDMABar = 0;
  throwExceptionIfOutOfBounds(dataToWrite, bytesToWrite);
  _mappedDevice->writeDMA(regOffset, mtca4upy::extractDataPointer(dataToWrite),
                          bytesToWrite, dummyDMABar);
}


void devMapAdapter::writeDMA(const std::string& regName,
                             bp::numeric::array dataToWrite, size_t dataSize,
                             uint32_t addRegOffset) {
  throwExceptionIfOutOfBounds(dataToWrite, dataSize);
  _mappedDevice->writeDMA(regName, mtca4upy::extractDataPointer(dataToWrite), dataSize,
                          addRegOffset);
}

devMapAdapter::devMapAdapter(mtca4u::devMap<mtca4u::devBase>* mappedDevice)
    : _mappedDevice(mappedDevice) {}

  mtca4u::devMap<mtca4u::devBase>::RegisterAccessor
  devMapAdapter::getRegisterAccessor (const std::string& regName) {
    return(_mappedDevice->getRegisterAccessor(regName));
  }

devMapAdapter::~devMapAdapter() {
  // TODO Auto-generated destructor stub
}
} /* namespace mtcapy */
