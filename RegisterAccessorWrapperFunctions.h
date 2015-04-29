#ifndef REGISTERACCESSORWRAPPERFUNCTIONS_H_
#define REGISTERACCESSORWRAPPERFUNCTIONS_H_

#include <MtcaMappedDevice/devMap.h>
#include "HelperFunctions.h"

namespace mtca4upy {
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

#endif /* REGISTERACCESSORWRAPPERFUNCTIONS_H_ */
