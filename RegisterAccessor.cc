#include "PythonModuleMethods.h"
#include "PythonExceptions.h"
#include <mtca4u/Utilities.h>

namespace mtca4upy {
namespace RegisterAccessor {
  // TODO: Reduce Boilerplate Code
  void read(mtca4u::Device::RegisterAccessor& self,
            bp::numeric::array& numpyArray, size_t numberOfElementsToRead,
            uint32_t startIndexInRegister) {

    if (extractDataType(numpyArray) == FLOAT64) {
      double* allocatedSpace =
          reinterpret_cast<double*>(extractDataPointer(numpyArray));
      self.read<double>(allocatedSpace, numberOfElementsToRead,
                       startIndexInRegister);
    } else {
      throw mtca4upy::ArrayElementTypeNotSupported(
          "Data format used is unsupported");
    }
  }

  void write(mtca4u::Device::RegisterAccessor& self,
             bp::numeric::array& numpyArray, size_t numElementsToWrite,
             uint32_t startIndexInRegister) {

    numpyDataTypes numpyArray_dType = extractDataType(numpyArray);
    char* dataToWrite = extractDataPointer(numpyArray);

    if (numpyArray_dType == INT32) {
      int* data = reinterpret_cast<int*>(dataToWrite);
      self.write(data, numElementsToWrite, startIndexInRegister);
    } else if (numpyArray_dType == INT64) {
      long int* data = reinterpret_cast<long int*>(dataToWrite);
      self.write(data, numElementsToWrite, startIndexInRegister);
    } else if (numpyArray_dType == FLOAT32) {
      float* data = reinterpret_cast<float*>(dataToWrite);
      self.write(data, numElementsToWrite, startIndexInRegister);
    } else if (numpyArray_dType == FLOAT64) {
      double* data = reinterpret_cast<double*>(dataToWrite);
      self.write(data, numElementsToWrite, startIndexInRegister);
    } else {
      throw mtca4upy::ArrayElementTypeNotSupported(
          "Data format used is unsupported");
    }
    /*
     TypeError: No registered converter was able to produce a C++ rvalue of
     type
     float from this Python object of type numpy.float32
   */
  }

  void readRaw(mtca4u::Device::RegisterAccessor& self,
               bp::numeric::array& numpyArray, size_t numberOfElementsToRead,
               uint32_t elementIndexInRegister) {

    if (extractDataType(numpyArray) == INT32) {
      int32_t* allocatedSpace =
          reinterpret_cast<int32_t*>(extractDataPointer(numpyArray));
      uint32_t byteOffsetFromRegBaseAddr =
          elementIndexInRegister * sizeof(uint32_t);
      size_t bytesToRead = numberOfElementsToRead * sizeof(uint32_t);
      self.readRaw(allocatedSpace, bytesToRead, byteOffsetFromRegBaseAddr);
    } else {
      throw mtca4upy::ArrayElementTypeNotSupported(
          "Data format used is unsupported");
    };
  }

  void writeRaw(mtca4u::Device::RegisterAccessor& self,
                bp::numeric::array& numpyArray, size_t numElementsToWrite,
                uint32_t elementIndexInRegister) {
    if (extractDataType(numpyArray) == INT32) {
      int32_t* userEnteredData =
          reinterpret_cast<int32_t*>(extractDataPointer(numpyArray));
      uint32_t byteOffsetFromRegBaseAddr =
          elementIndexInRegister * sizeof(uint32_t);
      size_t bytesToWrite = numElementsToWrite * sizeof(uint32_t);
      self.writeRaw(userEnteredData, bytesToWrite, byteOffsetFromRegBaseAddr);
    } else {
      throw mtca4upy::ArrayElementTypeNotSupported(
          "Data format used is unsupported");
    }
  }

  void readDMARaw(mtca4u::Device::RegisterAccessor& self,
                  bp::numeric::array& numpyArray, size_t numberOfElementsToRead,
                  uint32_t elementIndexInRegister) {

    if (extractDataType(numpyArray) == INT32) {
      int32_t* dataLocation =
          reinterpret_cast<int32_t*>(extractDataPointer(numpyArray));
      uint32_t dataOffset = elementIndexInRegister * sizeof(uint32_t);
      size_t dataSize = numberOfElementsToRead * sizeof(uint32_t);
      self.readDMA(dataLocation, dataSize, dataOffset);
    } else {
      throw mtca4upy::ArrayElementTypeNotSupported(
          "Data format used is unsupported");
    }
  }

  uint32_t size(mtca4u::Device::RegisterAccessor& self) {
    const mtca4u::RegisterInfoMap::RegisterInfo& registerDetails =
        self.getRegisterInfo();
    return (registerDetails.nElements);
  }
}
}

void mtca4upy::setDmapFile(const std::string& dmapFile) {
  mtca4u::setDMapFilePath(dmapFile);
}

std::string mtca4upy::getDmapFile() {
  return (mtca4u::getDMapFilePath());
}
