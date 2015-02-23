/*
 * DevBaseWrapper.h
 *
 *  Created on: Jan 28, 2015
 *      Author: varghese
 */

#ifndef SOURCE_DIRECTORY__DEVBASEWRAPPER_H_
#define SOURCE_DIRECTORY__DEVBASEWRAPPER_H_

#include <boost/python.hpp>
#include <MtcaMappedDevice/devBase.h>
#include <MtcaMappedDevice/devConfigBase.h>
#include "PythonInterface.h"

namespace mtca4upy { // TODO: Refactor to a better name

class PythonInterfaceWrapper
    : public mtca4upy::PythonInterface,
      public boost::python::wrapper<mtca4upy::PythonInterface> {
public:
  PythonInterfaceWrapper() {};
  inline void readDMA(uint32_t regOffset, bp::numeric::array Buffer,
                      size_t size) {
    this->get_override("readDMA")(regOffset, Buffer, size);
  }
  inline void writeDMA(uint32_t regOffset, bp::numeric::array dataToWrite,
                       size_t size) {

    this->get_override("writeDMA")(regOffset, dataToWrite, size);
  }

  inline void readArea(int32_t regOffset, bp::numeric::array Buffer,
                       size_t size, uint8_t bar) {
    this->get_override("readArea")(regOffset, Buffer, size, bar);
  }

  inline void writeArea(uint32_t regOffset, bp::numeric::array dataToWite,
                        size_t bytesToWrite, uint8_t bar) {
    this->get_override("writeArea")(regOffset, dataToWite, bytesToWrite, bar);
  }

  inline void readDMA(const std::string &regName,
                      bp::numeric::array bufferSpace, size_t dataSize = 0,
                      uint32_t addRegOffset = 0) {
    this->get_override("readDMA")(regName, bufferSpace, dataSize, addRegOffset);
  }
  inline void writeDMA(const std::string &regName,
                       bp::numeric::array dataToWrite, size_t dataSize = 0,
                       uint32_t addRegOffset = 0) {
    this->get_override("writeDMA")(regName, dataToWrite, dataSize,
                                   addRegOffset);
  }
  ~PythonInterfaceWrapper() {};
};
}
#endif /* SOURCE_DIRECTORY__DEVBASEWRAPPER_H_ */

