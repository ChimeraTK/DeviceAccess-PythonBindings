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

  inline void readRaw(uint32_t regOffset, bp::numeric::array Buffer,
                       size_t size, uint8_t bar) {
    this->get_override("readArea")(regOffset, Buffer, size, bar);
  }
  inline void readRaw(const std::string &regName, bp::numeric::array Buffer,
                      size_t dataSize, uint32_t addRegOffset){
    this->readRaw(regName, Buffer, dataSize, addRegOffset);
  }

  inline void writeRaw(uint32_t regOffset, bp::numeric::array dataToWite,
                        size_t bytesToWrite, uint8_t bar) {
    this->get_override("writeArea")(regOffset, dataToWite, bytesToWrite, bar);
  }

  inline void writeRaw(const std::string &regName, bp::numeric::array Buffer,
                        size_t dataSize, uint32_t addRegOffset){
    this->writeRaw(regName, Buffer, dataSize, addRegOffset);
  }

  inline void readDMA(uint32_t regOffset, bp::numeric::array Buffer,
                      size_t size) {
    this->get_override("readDMA")(regOffset, Buffer, size);
  }

  inline void writeDMA(uint32_t regOffset, bp::numeric::array dataToWrite,
                       size_t size) {

    this->get_override("writeDMA")(regOffset, dataToWrite, size);
  }

  inline void readDMA(const std::string &regName,
                      bp::numeric::array bufferSpace, size_t dataSize,
                      uint32_t addRegOffset) {
    this->get_override("readDMA")(regName, bufferSpace, dataSize, addRegOffset);
  }
  inline void writeDMA(const std::string &regName,
                       bp::numeric::array dataToWrite, size_t dataSize,
                       uint32_t addRegOffset) {
    this->get_override("writeDMA")(regName, dataToWrite, dataSize,
                                   addRegOffset);
  }

  inline mtca4u::devMap<mtca4u::devBase>::RegisterAccessor getRegisterAccessor(const std::string &regName){
    return(this->get_override("getRegisterAccessor")(regName));
  }
  ~PythonDeviceWrapper() {};
};
}
#endif /* SOURCE_DIRECTORY__DEVBASEWRAPPER_H_ */

