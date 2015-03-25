#ifndef SOURCE_DIRECTORY__DEVBASEWRAPPER_H_
#define SOURCE_DIRECTORY__DEVBASEWRAPPER_H_

#include <boost/python.hpp>
#include <MtcaMappedDevice/devBase.h>
#include <MtcaMappedDevice/devConfigBase.h>
#include "PythonDevice.h"

namespace mtca4upy { // TODO: Refactor to a better name

class PythonDeviceWrapper
    : public mtca4upy::PythonDevice,
      public boost::python::wrapper<mtca4upy::PythonDevice> {
public:
  PythonDeviceWrapper() {};

  inline void writeRaw(uint32_t regOffset, bp::numeric::array dataToWite,
                       size_t bytesToWrite, uint8_t bar) {
    this->get_override("writeArea")(regOffset, dataToWite, bytesToWrite, bar);
  }
  inline mtca4u::devMap<mtca4u::devBase>::RegisterAccessor getRegisterAccessor(
      const std::string &regName) {
    return (this->get_override("getRegisterAccessor")(regName));
  }
  ~PythonDeviceWrapper() {};
};
}
#endif /* SOURCE_DIRECTORY__DEVBASEWRAPPER_H_ */

