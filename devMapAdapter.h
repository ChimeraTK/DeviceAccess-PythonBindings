/*
 * devMapAdapter.h
 *
 *  Created on: Jan 30, 2015
 *      Author: varghese
 */

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

  void readDMA(uint32_t regOffset, bp::numeric::array Buffer, size_t size,
               uint8_t bar);
  void writeDMA(uint32_t regOffset, bp::numeric::array dataToWrite, size_t size,
                uint8_t bar);

  void readArea(int32_t regOffset, bp::numeric::array Buffer, size_t size,
                uint8_t bar);

  void writeArea(uint32_t regOffset, bp::numeric::array dataToWite,
                 size_t bytesToWrite, uint8_t bar);

  void readDMA(const std::string &regName, bp::numeric::array bufferSpace,
               size_t dataSize = 0, uint32_t addRegOffset = 0);
  void writeDMA(const std::string &regName, bp::numeric::array dataToWrite,
                size_t dataSize = 0, uint32_t addRegOffset = 0);

  ~devMapAdapter();
};

} /* namespace mtcapy */

#endif /* SOURCE_DIRECTORY__DEVMAPADAPTER_H_ */
