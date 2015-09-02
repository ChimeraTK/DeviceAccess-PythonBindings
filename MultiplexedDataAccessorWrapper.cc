#include "MultiplexedDataAccessorWrapper.h"

namespace mtca4upy {

MultiplexedDataAccessorWrapper::~MultiplexedDataAccessorWrapper() {
  // TODO Auto-generated destructor stub
}

} // namespace mtca4upy

mtca4upy::MultiplexedDataAccessorWrapper::MultiplexedDataAccessorWrapper(
    const boost::shared_ptr<mtca4u::devBase>& ioDevice,
    const std::vector<mtca4u::FixedPointConverter>& converters)
    : mtca4u::MultiplexedDataAccessor<float>(ioDevice, converters) {}

void mtca4upy::MultiplexedDataAccessorWrapper::read() {
  this->get_override("read")();
}

void mtca4upy::MultiplexedDataAccessorWrapper::write() {
  this->get_override("write")();
}

size_t mtca4upy::MultiplexedDataAccessorWrapper::getNumberOfDataSequences() {
  return (this->get_override("getNumberOfDataSequences")());
}
