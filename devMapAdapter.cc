#include "devMapAdapter.h"

namespace mtca4upy {

void devMapAdapter::writeRaw(uint32_t regOffset, bp::numeric::array dataToWrite,
                             size_t bytesToWrite, uint8_t bar) {
  throwExceptionIfOutOfBounds(dataToWrite, bytesToWrite);
  _mappedDevice->writeArea(regOffset, mtca4upy::extractDataPointer(dataToWrite),
                           bytesToWrite, bar);
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
