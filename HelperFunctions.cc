#include "HelperFunctions.h"
#include <numpy/arrayobject.h>
#include "PythonExceptions.h"

int32_t*
mtca4upy::extractDataPointer (const bp::numeric::array& Buffer)
			      {
  PyArrayObject* pointerToNumPyArrayMemory =
      reinterpret_cast<PyArrayObject*>(Buffer.ptr());
  return (reinterpret_cast<int32_t*>(pointerToNumPyArrayMemory->data));
}

void
mtca4upy::throwExceptionIfOutOfBounds (const bp::numeric::array& dataToWrite,
				       const size_t& bytesToWrite)
				       {
  size_t bytesInArray =
      (mtca4upy::extractNumberOfElements(dataToWrite) * sizeof(int32_t));
  if (bytesInArray < bytesToWrite) {
    throw mtca4upy::ArrayOutOfBoundException();
  }
}

size_t
mtca4upy::extractNumberOfElements (const bp::numeric::array& dataToWrite)
				   {
  return (boost::python::extract<long>(dataToWrite.attr("size")));
}
