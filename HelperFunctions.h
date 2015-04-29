#ifndef HELPERFUNCTIONS_H_
#define HELPERFUNCTIONS_H_
#include <boost/python.hpp>

namespace bp = boost::python;

namespace mtca4upy {

enum numpyDataTypes {
  INT32,
  INT64,
  FLOAT32,
  FLOAT64,
  USUPPORTED_TYPE
};

char *extractDataPointer(const bp::numeric::array &Buffer);

numpyDataTypes extractDataType(const bp::numeric::array &Buffer);

void throwExceptionIfOutOfBounds(const bp::numeric::array &dataToWrite,
                                 const size_t &bytesToWrite);

size_t extractNumberOfElements(const bp::numeric::array &dataToWrite);
}

#endif /* HELPERFUNCTIONS_H_ */
