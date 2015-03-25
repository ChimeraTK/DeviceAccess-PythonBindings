#include "RegisterAccessorWrapperFunctions.h"

// TODO: Reduce Boilerplate Code
void mtca4upy::readWrapper(
    mtca4u::devMap<mtca4u::devBase>::RegisterAccessor& self,
    bp::numeric::array& dataSpace, size_t arraySize,
    uint32_t elementIndexInRegister) {
  float* dataLocation = extractDataPointer<float>(dataSpace);
  uint32_t dataOffset = elementIndexInRegister *
                        sizeof(uint32_t); // This is assuming that the PCIE
                                          // mapped memory increments as a 32
                                          // bit word for each element
  self.read<float>(dataLocation, arraySize, dataOffset);
}

void mtca4upy::writeWrapper(
    mtca4u::devMap<mtca4u::devBase>::RegisterAccessor& self,
    bp::numeric::array& dataSpace, size_t arraySize,
    uint32_t elementIndexInRegister) {
  float* dataLocation = extractDataPointer<float>(dataSpace);
  uint32_t dataOffset = elementIndexInRegister * sizeof(uint32_t);
  self.write<float>(dataLocation, arraySize, dataOffset);
}

void mtca4upy::readRawWrapper(
    mtca4u::devMap<mtca4u::devBase>::RegisterAccessor& self,
    bp::numeric::array& dataSpace, size_t arraySize,
    uint32_t elementIndexInRegister) {
  int32_t* dataLocation = extractDataPointer<int32_t>(dataSpace);
  uint32_t dataOffset = elementIndexInRegister * sizeof(uint32_t);
  size_t dataSize = arraySize * sizeof(uint32_t);
  self.readReg(dataLocation, dataSize, dataOffset);
}

void mtca4upy::writeRawWrapper(
    mtca4u::devMap<mtca4u::devBase>::RegisterAccessor& self,
    bp::numeric::array& dataSpace, size_t arraySize,
    uint32_t elementIndexInRegister) {
  int32_t* dataLocation = extractDataPointer<int32_t>(dataSpace);
  uint32_t dataOffset = elementIndexInRegister * sizeof(uint32_t);
  size_t dataSize = arraySize * sizeof(uint32_t);
  self.writeReg(dataLocation, dataSize, dataOffset);
}

void mtca4upy::readDMARawWrapper(
    mtca4u::devMap<mtca4u::devBase>::RegisterAccessor& self,
    bp::numeric::array& dataSpace, size_t arraySize,
    uint32_t elementIndexInRegister) {
  int32_t* dataLocation = extractDataPointer<int32_t>(dataSpace);
  uint32_t dataOffset = elementIndexInRegister * sizeof(uint32_t);
  size_t dataSize = arraySize * sizeof(uint32_t);
  self.readDMA(dataLocation, dataSize, dataOffset);
}

uint32_t mtca4upy::sizeWrapper(
    mtca4u::devMap<mtca4u::devBase>::RegisterAccessor& self) {
  mtca4u::mapFile::mapElem mapelem = self.getRegisterInfo();
  return (mapelem.reg_elem_nr);
}
