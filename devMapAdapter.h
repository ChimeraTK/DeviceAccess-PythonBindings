#ifndef SOURCE_DIRECTORY__DEVMAPADAPTER_H_
#define SOURCE_DIRECTORY__DEVMAPADAPTER_H_

#include "PythonDevice.h"
#include <MtcaMappedDevice/devMap.h>
#include <MtcaMappedDevice/devBase.h>
#include <MtcaMappedDevice/DummyDevice.h>
#include <MtcaMappedDevice/devPCIE.h>

namespace mtca4upy {

class devMapAdapter : public PythonDevice {

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
   * This method lets the user write a block of data to the offset specified
   */
   void writeRaw(uint32_t regOffset, bp::numeric::array dataToWite,
                        size_t bytesToWrite, uint8_t bar);

  mtca4u::devMap<mtca4u::devBase>::RegisterAccessor getRegisterAccessor(const std::string &regName);

  void readDMA(uint32_t regOffset, bp::numeric::array Buffer, size_t size);
  void writeDMA(uint32_t regOffset, bp::numeric::array dataToWrite, size_t size);

  ~devMapAdapter();
};

} /* namespace mtcapy */

#endif /* SOURCE_DIRECTORY__DEVMAPADAPTER_H_ */
