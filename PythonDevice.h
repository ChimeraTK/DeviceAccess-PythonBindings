#ifndef SOURCE_DIRECTORY__PYTHONINTERFACE_H_
#define SOURCE_DIRECTORY__PYTHONINTERFACE_H_

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include "HelperFunctions.h"
#include "PythonExceptions.h"
#include <MtcaMappedDevice/devMap.h> // ideally should'nt be here; exclusive for acquiring register accessor

namespace mtca4upy {

/**
 * This is the supported interface on the python side. This class may be
 * inherited to create an adapter for any device that wants to hook into python.
 */
class PythonDevice {
public:
  PythonDevice() {};

  /**
   * This method lets the user write a block of data to the offset specified
   */
  virtual void writeRaw(uint32_t regOffset, bp::numeric::array dataToWite,
                        size_t bytesToWrite, uint8_t bar) = 0;

  virtual boost::shared_ptr<mtca4u::devMap<mtca4u::devBase>::RegisterAccessor>
  getRegisterAccessor(const std::string& moduleName,
                      const std::string& regName) = 0;

  virtual ~PythonDevice() {};
};

} /* namespace mtcapy */

#endif /* SOURCE_DIRECTORY__PYTHONINTERFACE_H_ */
