#include "HelperFunctions.h"
#include <numpy/arrayobject.h>

float*
mtca4upy::extractDataPointer (const bp::numeric::array& Buffer)
			      {
  PyArrayObject* pointerToNumPyArrayMemory =
      reinterpret_cast<PyArrayObject*>(Buffer.ptr());
  return (reinterpret_cast<float*>(pointerToNumPyArrayMemory->data));
}
