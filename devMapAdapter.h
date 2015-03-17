#ifndef SOURCE_DIRECTORY__DEVMAPADAPTER_H_
#define SOURCE_DIRECTORY__DEVMAPADAPTER_H_

#include "PythonInterface.h"
#include <MtcaMappedDevice/devMap.h>
#include <MtcaMappedDevice/devBase.h>
#include <MtcaMappedDevice/DummyDevice.h>
#include <MtcaMappedDevice/devPCIE.h>

namespace mtca4upy {

class devMapAdapter : public PythonInterface {

  boost::shared_ptr<mtca4u::devMap<mtca4u::devBase> > _mappedDevice; // TODO:
                                                                     // This has
                                                                     // to
                                                                     // change
                                                                     // when teh
                                                                     // devMAp
                                                                     // class is
                                                                     // changed
                                                                     // to a non
                                                                     // template
                                                                     // class
public:
  /*
   * Note: The mappedDevice has to be dynamically allocated and is expected to
   * be opened before being passed to the devMapAdapter
   */
  devMapAdapter(mtca4u::devMap<mtca4u::devBase> *mappedDevice); // This has to
                                                                // be revised
                                                                // once devMap
                                                                // class is
                                                                // changed to a
                                                                // non template
                                                                // class.



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
   * to it with the data in Buffer;
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

  ~devMapAdapter();
};

} /* namespace mtcapy */

#endif /* SOURCE_DIRECTORY__DEVMAPADAPTER_H_ */
