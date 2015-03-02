#include "PythonInterface.h"
#include <numpy/arrayobject.h>

namespace mtca4upy {

int32_t* PythonInterface::extractDataPointer(const bp::numeric::array& Buffer) {
  PyArrayObject* pointerToNumPyArrayMemory =
      reinterpret_cast<PyArrayObject*>(Buffer.ptr());
  return (reinterpret_cast<int32_t*>(pointerToNumPyArrayMemory->data));
}

size_t PythonInterface::extractNumberOfElements(
    const bp::numeric::array& dataToWrite) {
  return (boost::python::extract<long>(dataToWrite.attr("size")));
}

void mtca4upy::PythonInterface::throwExceptionIfOutOfBounds(
    const bp::numeric::array& dataToWrite, const size_t& bytesToWrite) {
  size_t bytesInArray =
      (PythonInterface::extractNumberOfElements(dataToWrite) * sizeof(int32_t));
  if (bytesInArray < bytesToWrite) {
    throw mtca4upy::ArrayOutOfBoundException();
  }
}

} /* namespace mtcapy */

