#ifndef _WRAPPER_METHODS_H_
#define _WRAPPER_METHODS_H_

#include <MtcaMappedDevice/devBase.h>
#include <boost/python.hpp>

namespace bp = boost::python;

namespace mtca4upy { // TODO: Refactor to a better name

  class ArrayOutOfBoundException: public std::exception{
  public:
    inline virtual const char* what() const throw()
     {
       return "size to write is more than the specified array size";
     }
  };

int32_t readReg(mtca4u::devBase &self, uint32_t registerOffset, uint8_t bar);
void writeReg(mtca4u::devBase &self, uint32_t regOffset, int32_t data,
              uint8_t bar);

void readDMA(mtca4u::devBase &self, uint32_t regOffset,
             bp::numeric::array Buffer, size_t size, uint8_t bar);
void readArea(mtca4u::devBase &self, int32_t regOffset,
              bp::numeric::array Buffer, size_t size, uint8_t bar);

void writeArea(mtca4u::devBase &self, uint32_t regOffset,
               bp::numeric::array dataToWite, size_t bytesToWrite, uint8_t bar);
std::string readDeviceInfo(mtca4u::devBase &self);

void openDev(mtca4u::devBase &self, const std::string &devName);
void closeDev(mtca4u::devBase &self);

// Helper Methods
int32_t *extractDataPointer(bp::numeric::array &Buffer);
size_t extractArraySize(bp::numeric::array &dataToWrite);

void translate(ArrayOutOfBoundException const& exception);


}
#endif /*_WRAPPER_METHODS_H_ */
