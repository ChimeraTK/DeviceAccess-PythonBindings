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
    throw mtca4upy::ArrayElementTypeNotSupported(
        "Data format used is unsupported");
  }
}

devMapAdapter::devMapAdapter(mtca4u::devMap<mtca4u::devBase>* mappedDevice)
    : _mappedDevice(mappedDevice) {}

boost::shared_ptr<mtca4u::devMap<mtca4u::devBase>::RegisterAccessor>
devMapAdapter::getRegisterAccessor(const std::string& moduleName,
                                   const std::string& regName) {
  return (_mappedDevice->getRegisterAccessor(regName, moduleName));
}

devMapAdapter::~devMapAdapter() {
  // TODO Auto-generated destructor stub
}
} /* namespace mtcapy */

boost::shared_ptr<mtca4u::MultiplexedDataAccessor<float> >
mtca4upy::devMapAdapter::getMultiplexedDataAccessor(
    const std::string& moduleName, const std::string& regionName) {
  return (
      _mappedDevice->getCustomAccessor<mtca4u::MultiplexedDataAccessor<float> >(
          regionName, moduleName));
}
