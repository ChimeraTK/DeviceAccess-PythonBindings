#include "WrapperMethods.h"
#include <vector>
#include <numpy/arrayobject.h>

namespace mtca4upy {
void openDev(mtca4u::devBase& self, const std::string& devName) {
  self.openDev(devName, O_RDWR, NULL);
}

int32_t readReg(mtca4u::devBase& self, uint32_t registerOffset, uint8_t bar) {
  int32_t registerContent;
  self.readReg(registerOffset, &registerContent, bar);
  return registerContent;
}

void writeReg(mtca4u::devBase& self, uint32_t regOffset, int32_t data,
              uint8_t bar) {
  self.writeReg(regOffset, data, bar);
}

std::string readDeviceInfo(mtca4u::devBase& self) {
  std::string deviceInfo;
  self.readDeviceInfo(&deviceInfo);
  return deviceInfo;
}

void closeDev(mtca4u::devBase& self) { self.closeDev(); }

void readDMA(mtca4u::devBase& self, uint32_t regOffset,
             bp::numeric::array Buffer, size_t size, uint8_t bar) {
  int32_t* pointerToMemory = extractDataPointer(Buffer);
  self.readDMA(regOffset, pointerToMemory, size, bar);
}

void readArea(mtca4u::devBase& self, int32_t regOffset,
              bp::numeric::array Buffer, size_t size, uint8_t bar) {
  int32_t* pointerToMemory = extractDataPointer(Buffer);
  self.readArea(regOffset, pointerToMemory, size, bar);
}

void writeArea(mtca4u::devBase& self, uint32_t regOffset,
               bp::numeric::array dataToWite, size_t bytesToWrite,
               uint8_t bar) {

  size_t bytesInArray = (extractNumberOfElements(dataToWite) * sizeof(int32_t));
  if (bytesInArray < bytesToWrite) {
    throw ArrayOutOfBoundException();
  }

  int32_t* pointerToMemory = extractDataPointer(dataToWite);
  self.writeArea(regOffset, pointerToMemory, bytesToWrite, bar);
}

// Helper Methods
int32_t* extractDataPointer(bp::numeric::array& Buffer) {
  PyArrayObject* pointerToNumPyArrayMemory =
      reinterpret_cast<PyArrayObject*>(Buffer.ptr());
  return (reinterpret_cast<int32_t*>(pointerToNumPyArrayMemory->data));
}

size_t extractNumberOfElements(bp::numeric::array& dataToWrite) {
  return (boost::python::extract<long>(dataToWrite.attr("size")));
}
}
