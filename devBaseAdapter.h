#ifndef DEVBASEADAPTER_H_
#define DEVBASEADAPTER_H_

#include "PythonDevice.h"
#include <MtcaMappedDevice/devBase.h>

namespace mtca4upy {

/**
 * This class makes the interface of the mtca4u::devBase class compatible with
 * the supported python interface. _mtcaDevice is a pointer to a dynamically
 * allocated version of a devBase device. It is expected to open the _mtcaDevice
 * before passing it into the devBaseAdapter
 */
class devBaseAdapter : public mtca4upy::PythonDevice {
  boost::shared_ptr<mtca4u::devBase> _mtcaDevice;

public:
  /*
   * Note: mtcaDevice has to be dynamically allocated and opened before passing
   * it into the devBaseAdapter
   */
  devBaseAdapter(mtca4u::devBase *mtcaDevice);

  /**
   * This method lets the user write a block of data to the offset specified
   */
   void writeRaw(uint32_t regOffset, bp::numeric::array dataToWite,
                        size_t bytesToWrite, uint8_t bar);

   mtca4u::devMap<mtca4u::devBase>::RegisterAccessor getRegisterAccessor(const std::string &regName);


  ~devBaseAdapter();
};

} /* namespace mtcapy */

#endif /* DEVBASEADAPTER_H_ */
