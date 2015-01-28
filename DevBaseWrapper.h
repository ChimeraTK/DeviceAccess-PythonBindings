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

namespace mtca4upy { // TODO: Refactor to a better name

class DevBaseWrapper : public mtca4u::devBase,
                       public boost::python::wrapper<mtca4u::devBase> {
public:
  DevBaseWrapper();
  // Right now the methods are just stubs with no implementation.
  void openDev(const std::string& devName, int perm = O_RDWR,
               mtca4u::devConfigBase* pConfig = NULL);
  void closeDev();
  void readReg(uint32_t regOffset, int32_t* data, uint8_t bar);
  void writeReg(uint32_t regOffset, int32_t data, uint8_t bar);
  void readArea(uint32_t regOffset, int32_t* data, size_t size, uint8_t bar);
  void writeArea(uint32_t regOffset, int32_t const* data, size_t size,
                 uint8_t bar);
  void readDMA(uint32_t regOffset, int32_t* data, size_t size, uint8_t bar);
  void writeDMA(uint32_t regOffset, int32_t const* data, size_t size,
                uint8_t bar);
  void readDeviceInfo(std::string* devInfo);
  bool isOpen();
  ~DevBaseWrapper();
};
}
#endif /* SOURCE_DIRECTORY__DEVBASEWRAPPER_H_ */

