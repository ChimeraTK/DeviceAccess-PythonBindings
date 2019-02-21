#ifndef HELPERFUNCTIONS_H_
#define HELPERFUNCTIONS_H_
#include "NumpyObjectManager.h"
#include "PythonExceptions.h"
#include <boost/python.hpp>
#include <numpy/arrayobject.h>

namespace bp = boost::python;

namespace mtca4upy {

enum numpyDataTypes { INT32, INT64, FLOAT32, FLOAT64, USUPPORTED_TYPE };

inline size_t
extractNumberOfElements(const mtca4upy::NumpyObject &dataToWrite) {
  return (boost::python::extract<long>(dataToWrite.attr("size")));
}

inline char *extractDataPointer(const mtca4upy::NumpyObject &Buffer) {
  PyArrayObject *numpyArrayObj =
      reinterpret_cast<PyArrayObject *>(Buffer.ptr());
  return PyArray_BYTES(numpyArrayObj);
}

inline void
throwExceptionIfOutOfBounds(const mtca4upy::NumpyObject &dataToWrite,
                            const size_t &bytesToWrite) {
  size_t bytesInArray =
      (extractNumberOfElements(dataToWrite) * sizeof(int32_t));
  if (bytesInArray < bytesToWrite) {
    throw mtca4upy::ArrayOutOfBoundException(
        "size to write is more than the supplied array size");
  }
}

inline mtca4upy::numpyDataTypes
extractDataType(const mtca4upy::NumpyObject &Buffer) {
  PyArrayObject *numpyArrayObj =
      reinterpret_cast<PyArrayObject *>(Buffer.ptr());
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
} // namespace mtca4upy

#endif /* HELPERFUNCTIONS_H_ */
