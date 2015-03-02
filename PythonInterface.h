#ifndef SOURCE_DIRECTORY__PYTHONINTERFACE_H_
#define SOURCE_DIRECTORY__PYTHONINTERFACE_H_

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include "PythonExceptions.h"

namespace bp = boost::python; // TODO: remove this definition from other files
                              // once this has been derived and used in
                              // MtcaMappedDevicePython.cc

namespace mtca4upy {

/**
 * This is the supported interface on the python side. This class may be
 * inherited to create an adapter for any device that wants to hook into python.
 */
class PythonInterface {
public:
  PythonInterface() {};

  /**
   * Intent of this method is to read a block of specified size starting from
   * the specified address offset, on the specified address bar
   */
  virtual void readRaw(uint32_t regOffset, bp::numeric::array Buffer,
                       size_t size, uint8_t bar) = 0;

  /**
   * This method searches for the register specified in 'regName' and returns
   * its contents. Right now responsibility is on the user to pass in a 'Buffer'
   * that is large enough to accommodate contents of the register
   */
  virtual void readRaw(const std::string &regName, bp::numeric::array Buffer,
                       size_t dataSize, uint32_t addRegOffset) = 0;

  /**
   * This method lets the user write a block of data to the offset specified
   */
  virtual void writeRaw(uint32_t regOffset, bp::numeric::array dataToWite,
                        size_t bytesToWrite, uint8_t bar) = 0;

  /**
   * This method searches for the register name in regName and if found, writes
   * to it with the data in Buffer; TODO: Bad part is user has to be aware of
   * the reg size + have to prevent user from writing out of bounds
   */
  virtual void writeRaw(const std::string &regName, bp::numeric::array Buffer,
                        size_t dataSize, uint32_t addRegOffset) = 0;



  virtual void readDMA(uint32_t regOffset, bp::numeric::array Buffer,
                       size_t size) = 0;
  virtual void writeDMA(uint32_t regOffset, bp::numeric::array dataToWrite,
                        size_t size) = 0;

  virtual void readDMA(const std::string &regName,
                       bp::numeric::array bufferSpace, size_t dataSize,
                       uint32_t addRegOffset) = 0;
  virtual void writeDMA(const std::string &regName,
                        bp::numeric::array dataToWrite, size_t dataSize,
                        uint32_t addRegOffset) = 0;

  virtual ~PythonInterface() {};

protected:
  void throwExceptionIfOutOfBounds(const bp::numeric::array &dataToWrite,
                                   const size_t &bytesToWrite);
  // Helper functions for conversions on bp::numeric::array
  int32_t *extractDataPointer(const bp::numeric::array &Buffer);
  size_t extractNumberOfElements(const bp::numeric::array &dataToWrite);
};

} /* namespace mtcapy */

#endif /* SOURCE_DIRECTORY__PYTHONINTERFACE_H_ */
