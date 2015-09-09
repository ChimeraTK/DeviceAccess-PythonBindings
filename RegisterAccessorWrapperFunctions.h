#ifndef REGISTERACCESSORWRAPPERFUNCTIONS_H_
#define REGISTERACCESSORWRAPPERFUNCTIONS_H_

#include <MtcaMappedDevice/devMap.h>
#include "HelperFunctions.h"

// FIXME: extract 'size_t arraySize' from  bp::numeric array.

namespace mtca4upy {
namespace RegisterAccessor {

void readWrapper(mtca4u::devMap<mtca4u::devBase>::RegisterAccessor &self,
                 bp::numeric::array &numpyArray, size_t arraySize,
                 uint32_t elementIndexInRegister);
void writeWrapper(mtca4u::devMap<mtca4u::devBase>::RegisterAccessor &self,
                  bp::numeric::array &numpyArray, size_t numElements,
                  uint32_t elementIndexInRegister);
void readRawWrapper(mtca4u::devMap<mtca4u::devBase>::RegisterAccessor &self,
                    bp::numeric::array &numpyArray, size_t arraySize,
                    uint32_t elementIndexInRegister);
void writeRawWrapper(mtca4u::devMap<mtca4u::devBase>::RegisterAccessor &self,
                     bp::numeric::array &numpyArray, size_t arraySize,
                     uint32_t elementIndexInRegister);
void readDMARawWrapper(mtca4u::devMap<mtca4u::devBase>::RegisterAccessor &self,
                       bp::numeric::array &numpyArray, size_t arraySize,
                       uint32_t elementIndexInRegister);
uint32_t sizeWrapper(mtca4u::devMap<mtca4u::devBase>::RegisterAccessor &self);
}

namespace MuxDataAccessor {

void readInDataFromCard(mtca4u::MultiplexedDataAccessor<float> &self);
size_t getSequenceCount(mtca4u::MultiplexedDataAccessor<float> &self);
size_t getBlockCount(mtca4u::MultiplexedDataAccessor<float> &self);
void copyReadInData(mtca4u::MultiplexedDataAccessor<float> &self,
          bp::numeric::array &numpyArray);

} // namespace RegisterAccessor
} // namespace mtca4upy

#endif /* REGISTERACCESSORWRAPPERFUNCTIONS_H_ */
