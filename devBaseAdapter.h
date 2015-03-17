#ifndef DEVBASEADAPTER_H_
#define DEVBASEADAPTER_H_

#include "PythonInterface.h"
#include <MtcaMappedDevice/devBase.h>

namespace mtca4upy {

/**
 * This class makes the interface of the mtca4u::devBase class compatible with
 * the supported python interface. _mtcaDevice is a pointer to a dynamically
 * allocated version of a devBase device. It is expected to open the _mtcaDevice
 * before passing it into the devBaseAdapter
 */
class devBaseAdapter : public mtca4upy::PythonInterface {
  boost::shared_ptr<mtca4u::devBase> _mtcaDevice;

public:
  /*
   * Note: mtcaDevice has to be dynamically allocated and opened before passing
   * it into the devBaseAdapter
   */
  devBaseAdapter(mtca4u::devBase *mtcaDevice);


  /**
   * Intent of this method is to read a block of specified size starting from
   * the specified address offset, on the specified address bar
   */
   void readRaw(uint32_t regOffset, bp::numeric::array Buffer,
                       size_t size, uint8_t bar);

  /**
   * This method searches for the register specified in 'regName' and returns
   * its contents. Right now responsibility is on the user to pass in a 'Buffer'
   * that is large enough to accommodate contents of the register
   */
   void readRaw(const std::string &regName, bp::numeric::array Buffer,
                       size_t dataSize = 0, uint32_t addRegOffset = 0);

  /**
   * This method lets the user write a block of data to the offset specified
   */
   void writeRaw(uint32_t regOffset, bp::numeric::array dataToWite,
                        size_t bytesToWrite, uint8_t bar);

  /**
   * This method searches for the register name in regName and if found, writes
   * to it with the data in Buffer; TODO: Bad part is user has to be aware of
   * the reg size + have to prevent user from writing out of bounds
   */
   void writeRaw(const std::string &regName, bp::numeric::array Buffer,
                        size_t dataSize = 0, uint32_t addRegOffset = 0);


   mtca4u::devMap<mtca4u::devBase>::RegisterAccessor getRegisterAccessor(const std::string &regName);

  void readDMA(uint32_t regOffset, bp::numeric::array Buffer, size_t size);
  void writeDMA(uint32_t regOffset, bp::numeric::array dataToWrite, size_t size);
  void readDMA(const std::string &regName, bp::numeric::array bufferSpace,
               size_t dataSize = 0, uint32_t addRegOffset = 0);
  void writeDMA(const std::string &regName, bp::numeric::array dataToWrite,
                size_t dataSize = 0, uint32_t addRegOffset = 0);

  ~devBaseAdapter();
};

} /* namespace mtcapy */

#endif /* DEVBASEADAPTER_H_ */
