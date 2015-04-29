#include "HelperFunctions.h"
#include <numpy/arrayobject.h>
#include "PythonExceptions.h"

char* mtca4upy::extractDataPointer(const bp::numeric::array& Buffer) {
  PyArrayObject* numpyArrayObj = reinterpret_cast<PyArrayObject*>(Buffer.ptr());
  return (numpyArrayObj->data);
}

void mtca4upy::throwExceptionIfOutOfBounds(
    const bp::numeric::array& dataToWrite, const size_t& bytesToWrite) {
  size_t bytesInArray =
      (mtca4upy::extractNumberOfElements(dataToWrite) * sizeof(int32_t));
  if (bytesInArray < bytesToWrite) {
    throw mtca4upy::ArrayOutOfBoundException();
  }
}

mtca4upy::numpyDataTypes mtca4upy::extractDataType(
    const bp::numeric::array& Buffer) {

  PyArrayObject* numpyArrayObj = reinterpret_cast<PyArrayObject*>(Buffer.ptr());
  int type_number = PyArray_TYPE(numpyArrayObj);

  if (type_number == NPY_INT32) {
    return INT32;
  } else if (type_number == NPY_INT64) {
    return INT64;
  } else if (type_number == NPY_FLOAT32) {
    return FLOAT32;
  } else if (type_number == NPY_FLOAT64) {
    return FLOAT64;
  } else {
    return USUPPORTED_TYPE;
  }
}

size_t mtca4upy::extractNumberOfElements(
    const bp::numeric::array& dataToWrite) {
  return (boost::python::extract<long>(dataToWrite.attr("size")));
}
