#include "HelperFunctions.h"
#include <numpy/arrayobject.h>
#include "PythonExceptions.h"

char* mtca4upy::extractDataPointer(const bp::numeric::array& Buffer) {
  PyArrayObject* pointerToNumPyArrayMemory =
      reinterpret_cast<PyArrayObject*>(Buffer.ptr());
  return (pointerToNumPyArrayMemory->data);
}


void mtca4upy::throwExceptionIfOutOfBounds(
    const bp::numeric::array& dataToWrite, const size_t& bytesToWrite) {
  size_t bytesInArray =
      (mtca4upy::extractNumberOfElements(dataToWrite) * sizeof(int32_t));
  if (bytesInArray < bytesToWrite) {
    throw mtca4upy::ArrayOutOfBoundException();
  }
}

mtca4upy::numpyArrayWordSize mtca4upy::extractWordSizeInArray (const bp::numeric::array& Buffer){

  PyArrayObject* pointerToNumPyArrayMemory =
    reinterpret_cast<PyArrayObject*>(Buffer.ptr());
 int wordSizeInBytes = *(pointerToNumPyArrayMemory->strides);

 if (wordSizeInBytes == sizeof(uint64_t)){
   return SIZE_64_BITS;
 } else if (wordSizeInBytes == sizeof(uint32_t)) {
     return SIZE_32_BITS;
 } else if (wordSizeInBytes == sizeof(uint8_t)) {
     return SIZE_8_BITS;
 } else if (wordSizeInBytes == sizeof(uint16_t)) {
     return SIZE_16_BITS;
 } else {
     return USUPPORTED_SIZE;
 }
}

size_t mtca4upy::extractNumberOfElements(
    const bp::numeric::array& dataToWrite) {
  return (boost::python::extract<long>(dataToWrite.attr("size")));
}
