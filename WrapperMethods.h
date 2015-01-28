#ifndef _WRAPPER_METHODS_H_
#define _WRAPPER_METHODS_H_

#include <MtcaMappedDevice/devPCIE.h>
#include <boost/python.hpp>

namespace mtca4upy { // TODO: Refactor to a better name
int32_t readReg(mtca4u::devPCIE &self, uint32_t registerOffset, uint8_t bar);

/*
 * parameter size is the size in bytes, that the method has to return
 *
 */
boost::python::list readDMA(mtca4u::devPCIE &self, uint32_t regOffset,
                            size_t size, uint8_t bar);
boost::python::list readArea(mtca4u::devPCIE &self, int32_t regOffset,
                             size_t size, uint8_t bar);
void writeArea(mtca4u::devPCIE &self, uint32_t regOffset,
               boost::python::list data, size_t size, uint8_t bar);
std::string readDeviceInfo(mtca4u::devPCIE &self);

void openDev(mtca4u::devPCIE &self, const std::string &devName);
}
#endif /*_WRAPPER_METHODS_H_ */
