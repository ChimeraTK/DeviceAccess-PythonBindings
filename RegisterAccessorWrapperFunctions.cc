#include "RegisterAccessorWrapperFunctions.h"
#include "PythonExceptions.h"
#include "MtcaMappedDevice/FixedPointConverter.h"

// TODO: Reduce Boilerplate Code
void mtca4upy::RegisterAccessor::readWrapper(
    mtca4u::devMap<mtca4u::devBase>::RegisterAccessor& self,
    bp::numeric::array& numpyArray, size_t arraySize,
    uint32_t elementIndexInRegister) {

  if (extractDataType(numpyArray) == FLOAT32) {
    float* dataLocation =
        reinterpret_cast<float*>(extractDataPointer(numpyArray));
    self.read<float>(dataLocation, arraySize, elementIndexInRegister);
  } else {
    throw mtca4upy::ArrayElementTypeNotSupported(
        "Data format used is unsupported");
  }
}

void mtca4upy::RegisterAccessor::writeWrapper(
    mtca4u::devMap<mtca4u::devBase>::RegisterAccessor& self,
    bp::numeric::array& numpyArray, size_t numElements,
    uint32_t elementIndexInRegister) {

  std::vector<int32_t> rawData(numElements);
  uint32_t offsetInBytes = elementIndexInRegister * sizeof(uint32_t);
  const mtca4u::FixedPointConverter& fPConvrter = self.getFixedPointConverter();

  numpyDataTypes dTypeNumpyArray = extractDataType(numpyArray);
  char* dataPointerInArray = extractDataPointer(numpyArray);

  if (dTypeNumpyArray == INT32) {
    int* data = reinterpret_cast<int*>(dataPointerInArray);
    for (size_t i = 0; i < numElements; ++i) {
      rawData[i] = fPConvrter.toFixedPoint(data[i]);
    }
  } else if (dTypeNumpyArray == INT64) {
    long int* data = reinterpret_cast<long int*>(dataPointerInArray);
    for (size_t i = 0; i < numElements; ++i) {
      rawData[i] = fPConvrter.toFixedPoint(data[i]);
    }
  } else if (dTypeNumpyArray == FLOAT32) {
    float* data = reinterpret_cast<float*>(dataPointerInArray);
    for (size_t i = 0; i < numElements; ++i) {
      rawData[i] = fPConvrter.toFixedPoint(data[i]);
    }
  } else if (dTypeNumpyArray == FLOAT64) {
    double* data = reinterpret_cast<double*>(dataPointerInArray);
    for (size_t i = 0; i < numElements; ++i) {
      rawData[i] = fPConvrter.toFixedPoint(data[i]);
    }
  } else {
    throw mtca4upy::ArrayElementTypeNotSupported(
        "Data format used is unsupported");
  }

  self.writeReg(&(rawData[0]), numElements * sizeof(int32_t), offsetInBytes);

  /*
   TypeError: No registered converter was able to produce a C++ rvalue of type
   float from this Python object of type numpy.float32
 */
}

void mtca4upy::RegisterAccessor::readRawWrapper(
    mtca4u::devMap<mtca4u::devBase>::RegisterAccessor& self,
    bp::numeric::array& numpyArray, size_t arraySize,
    uint32_t elementIndexInRegister) {

  if (extractDataType(numpyArray) == INT32) {
    int32_t* dataLocation =
        reinterpret_cast<int32_t*>(extractDataPointer(numpyArray));
    uint32_t dataOffset = elementIndexInRegister * sizeof(uint32_t);
    size_t dataSize = arraySize * sizeof(uint32_t);
    self.readReg(dataLocation, dataSize, dataOffset);
  } else {
    throw mtca4upy::ArrayElementTypeNotSupported(
        "Data format used is unsupported");
  };
}

void mtca4upy::RegisterAccessor::writeRawWrapper(
    mtca4u::devMap<mtca4u::devBase>::RegisterAccessor& self,
    bp::numeric::array& numpyArray, size_t arraySize,
    uint32_t elementIndexInRegister) {
  if (extractDataType(numpyArray) == INT32) {
    int32_t* dataLocation =
        reinterpret_cast<int32_t*>(extractDataPointer(numpyArray));
    uint32_t dataOffset = elementIndexInRegister * sizeof(uint32_t);
    size_t dataSize = arraySize * sizeof(uint32_t);
    self.writeReg(dataLocation, dataSize, dataOffset);
  } else {
    throw mtca4upy::ArrayElementTypeNotSupported(
        "Data format used is unsupported");
  }
}

void mtca4upy::RegisterAccessor::readDMARawWrapper(
    mtca4u::devMap<mtca4u::devBase>::RegisterAccessor& self,
    bp::numeric::array& numpyArray, size_t arraySize,
    uint32_t elementIndexInRegister) {

  if (extractDataType(numpyArray) == INT32) {
    int32_t* dataLocation =
        reinterpret_cast<int32_t*>(extractDataPointer(numpyArray));
    uint32_t dataOffset = elementIndexInRegister * sizeof(uint32_t);
    size_t dataSize = arraySize * sizeof(uint32_t);
    self.readDMA(dataLocation, dataSize, dataOffset);
  } else {
    throw mtca4upy::ArrayElementTypeNotSupported(
        "Data format used is unsupported");
  }
}

uint32_t mtca4upy::RegisterAccessor::sizeWrapper(
    mtca4u::devMap<mtca4u::devBase>::RegisterAccessor& self) {
  const mtca4u::mapFile::mapElem& mapelem = self.getRegisterInfo();
  return (mapelem.reg_elem_nr);
}

void mtca4upy::MuxDataAccessor::readInDataFromCard(
    mtca4u::MultiplexedDataAccessor<float>& self) {
  self.read();
}

size_t mtca4upy::MuxDataAccessor::getSequenceCount(
    mtca4u::MultiplexedDataAccessor<float>& self) {
  return (self.getNumberOfDataSequences());
}

size_t mtca4upy::MuxDataAccessor::getBlockCount(
    mtca4u::MultiplexedDataAccessor<float>& self) {
  // FIXME: Make sure prepareAccessor was called. check and throw an exception
  // if
  // it was not
  return (self[0].size());
}

void mtca4upy::MuxDataAccessor::copyReadInData(
    mtca4u::MultiplexedDataAccessor<float>& self,
    bp::numeric::array& numpyArray) {
  // FIXME: Make sure prepareAccessor was called. check and throw an exception
  // if
  // it was not
  float* data = reinterpret_cast<float*>(extractDataPointer(numpyArray));

  size_t numSequences = self.getNumberOfDataSequences();
  size_t numBlocks = self[0].size();

  size_t pyArrayRowStride = numSequences;
  for (size_t pyArrayCol = 0; pyArrayCol < numSequences; ++pyArrayCol) {
    for (size_t pyArrrayRow = 0; pyArrrayRow < numBlocks; ++pyArrrayRow) {
      // pyArrayCol corresponds to the sequence numbers and pyArrrayRow to each
      // element of the sequence
      data[(pyArrayRowStride * pyArrrayRow) + pyArrayCol] = self[pyArrayCol][pyArrrayRow];
    }
  }
}
