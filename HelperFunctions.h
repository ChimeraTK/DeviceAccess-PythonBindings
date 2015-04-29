#ifndef HELPERFUNCTIONS_H_
#define HELPERFUNCTIONS_H_
#include <boost/python.hpp>

namespace bp = boost::python;

namespace mtca4upy {

enum numpyArrayWordSize{
  SIZE_8_BITS,
  SIZE_16_BITS,
  SIZE_32_BITS,
  SIZE_64_BITS,
  USUPPORTED_SIZE
};

char* extractDataPointer(const bp::numeric::array &Buffer);

numpyArrayWordSize extractWordSizeInArray(const bp::numeric::array &Buffer);

void throwExceptionIfOutOfBounds(const bp::numeric::array &dataToWrite,
                                 const size_t &bytesToWrite);

size_t extractNumberOfElements(const bp::numeric::array &dataToWrite);

}

#endif /* HELPERFUNCTIONS_H_ */
