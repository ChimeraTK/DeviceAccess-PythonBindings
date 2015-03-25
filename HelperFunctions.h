#ifndef HELPERFUNCTIONS_H_
#define HELPERFUNCTIONS_H_
#include <boost/python.hpp>

namespace bp = boost::python;

namespace mtca4upy {

/**
 * Careful when using this. Ensure that type 'T' is a 32 bit datatype
 * TODO: Refactor to something safer/hard to abuse
 * TODO: fix / Write up new documentation for the helper functions
 */
template <typename T> T *extractDataPointer(const bp::numeric::array &Buffer);

void throwExceptionIfOutOfBounds(const bp::numeric::array &dataToWrite,
                                 const size_t &bytesToWrite);

size_t extractNumberOfElements(const bp::numeric::array &dataToWrite);
}

#endif /* HELPERFUNCTIONS_H_ */
