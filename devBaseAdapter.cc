#include "devBaseAdapter.h"

namespace mtca4upy {

devBaseAdapter::devBaseAdapter(mtca4u::devBase* mtcaDevice)
    : _mtcaDevice(mtcaDevice) {}

void devBaseAdapter::writeRaw(uint32_t regOffset,
                              bp::numeric::array dataToWrite,
                              size_t bytesToWrite, uint8_t bar) {
  throwExceptionIfOutOfBounds(dataToWrite, bytesToWrite);
  _mtcaDevice->writeArea(regOffset, extractDataPointer<int32_t>(dataToWrite),
                         bytesToWrite, bar);
}

mtca4u::devMap<mtca4u::devBase>::RegisterAccessor
devBaseAdapter::getRegisterAccessor(const std::string& regName) {
  (void)(regName);
  throw MethodNotImplementedException();
}

devBaseAdapter::~devBaseAdapter() {
  // TODO Auto-generated destructor stub
}

} /* namespace mtcapy */

