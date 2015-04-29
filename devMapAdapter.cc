#include "devMapAdapter.h"

namespace mtca4upy {

void devMapAdapter::writeRaw(uint32_t regOffset, bp::numeric::array dataToWrite,
                             size_t bytesToWrite, uint8_t bar) {
  throwExceptionIfOutOfBounds(dataToWrite, bytesToWrite);
  if (extractDataType(dataToWrite) == INT32) {
    int32_t* dataPointer =
        reinterpret_cast<int32_t*>(extractDataPointer(dataToWrite));
    _mappedDevice->writeArea(regOffset, dataPointer, bytesToWrite, bar);
  } else {
    throw mtca4upy::ArrayElementTypeNotSupported();
  }
}

devMapAdapter::devMapAdapter(mtca4u::devMap<mtca4u::devBase>* mappedDevice)
    : _mappedDevice(mappedDevice) {}

mtca4u::devMap<mtca4u::devBase>::RegisterAccessor
devMapAdapter::getRegisterAccessor(const std::string& regName) {
  return (_mappedDevice->getRegisterAccessor(regName));
}

devMapAdapter::~devMapAdapter() {
  // TODO Auto-generated destructor stub
}
} /* namespace mtcapy */
