/*
 * SimpleFactoty.h
 *
 *  Created on: Jan 28, 2015
 *      Author: varghese
 */

#ifndef SOURCE_DIRECTORY__SIMPLEFACTORY_H_
#define SOURCE_DIRECTORY__SIMPLEFACTORY_H_

#include <boost/shared_ptr.hpp>
#include <MtcaMappedDevice/devBase.h>

namespace mtca4upy { // TODO: Refactor to a better name
class Device {
public:
  Device();
  boost::shared_ptr<mtca4u::devBase> createPCIEDevice();
  boost::shared_ptr<mtca4u::devBase> createDummyDevice();
  ~Device();
};
}
#endif /* SOURCE_DIRECTORY__SIMPLEFACTORY_H_ */
