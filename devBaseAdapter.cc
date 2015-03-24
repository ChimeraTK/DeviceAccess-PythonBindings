#include "devBaseAdapter.h"


namespace mtca4upy {

devBaseAdapter::devBaseAdapter(mtca4u::devBase* mtcaDevice)
    : _mtcaDevice(mtcaDevice) {}

void devBaseAdapter::readRaw(uint32_t regOffset, bp::numeric::array Buffer,
                              size_t size, uint8_t bar) {
  throwExceptionIfOutOfBounds(Buffer, size);
  _mtcaDevice->readArea(regOffset, extractDataPointer(Buffer), size, bar);
}

void devBaseAdapter::writeRaw(uint32_t regOffset,
                               bp::numeric::array dataToWrite,
                               size_t bytesToWrite, uint8_t bar) {
  throwExceptionIfOutOfBounds(dataToWrite, bytesToWrite);
  _mtcaDevice->writeArea(regOffset, extractDataPointer(dataToWrite),
                         bytesToWrite, bar);
}

void devBaseAdapter::readDMA(uint32_t regOffset, bp::numeric::array Buffer,
                             size_t size) {
  uint8_t dummyDMABar = 0; // This can be anything; DMA region is not dependent on the Pcie bar
  throwExceptionIfOutOfBounds(Buffer, size);
  _mtcaDevice->readDMA(regOffset, extractDataPointer(Buffer), size, dummyDMABar);
}

void devBaseAdapter::writeDMA(uint32_t regOffset,
                              bp::numeric::array dataToWrite, size_t size) {
  uint8_t dummyDMABar = 0; // This can be anything; DMA region is not dependent on the Pcie bar
  throwExceptionIfOutOfBounds(dataToWrite, size);
  _mtcaDevice->writeDMA(regOffset, extractDataPointer(dataToWrite), size, dummyDMABar);
}


void devBaseAdapter::writeDMA(const std::string& regName,
                              bp::numeric::array dataToWrite, size_t dataSize,
                              uint32_t addRegOffset) {
  // Workaround to compiler warning: unused parameter
  (void)(regName);
  (void)(dataToWrite);
  (void)(dataSize);
  (void)(addRegOffset);

  throw MethodNotImplementedException();
}

  mtca4u::devMap<mtca4u::devBase>::RegisterAccessor
  devBaseAdapter::getRegisterAccessor (const std::string& regName) {
    (void)(regName);
    throw MethodNotImplementedException();
  }

devBaseAdapter::~devBaseAdapter() {
  // TODO Auto-generated destructor stub
}

} /* namespace mtcapy */


