#include "HelperFunctions.h"
#include <numpy/arrayobject.h>
#include "PythonExceptions.h"

/**
 * Careful when using this. Ensure that type 'T' is a 32 bit datatype
 */
template <typename T>
T* mtca4upy::extractDataPointer(const bp::numeric::array& Buffer) {
  PyArrayObject* pointerToNumPyArrayMemory =
      reinterpret_cast<PyArrayObject*>(Buffer.ptr());
  return (reinterpret_cast<T*>(pointerToNumPyArrayMemory->data));
}

// Explicit instantiation to work around undefined symbol error in the compiled
// shared library
template int32_t* mtca4upy::extractDataPointer<int32_t>(
    const bp::numeric::array& Buffer);
template float* mtca4upy::extractDataPointer<float>(
    const bp::numeric::array& Buffer);

void mtca4upy::throwExceptionIfOutOfBounds(
    const bp::numeric::array& dataToWrite, const size_t& bytesToWrite) {
  size_t bytesInArray =
      (mtca4upy::extractNumberOfElements(dataToWrite) * sizeof(int32_t));
  if (bytesInArray < bytesToWrite) {
    throw mtca4upy::ArrayOutOfBoundException();
  }
}

size_t mtca4upy::extractNumberOfElements(
    const bp::numeric::array& dataToWrite) {
  return (boost::python::extract<long>(dataToWrite.attr("size")));
}

