#ifndef DEVICEFACTORY_H_
#define DEVICEFACTORY_H_

#include <boost/shared_ptr.hpp>
#include "PythonDevice.h"

namespace mtca4upy {

class DeviceInformation {
public:
  std::string deviceName;
  std::string deviceMapFileLocation;
  DeviceInformation(std::string aliasName, std::string mapFile);
};

/**
 * This abstract base class may be inherited to create a concrete factory.
 */
class DeviceFactory {
protected:
  boost::shared_ptr<mtca4upy::DeviceInformation> _deviceDetails;

public:
  DeviceFactory(mtca4upy::DeviceInformation* deviceDetails);
  virtual boost::shared_ptr<mtca4upy::PythonDevice> createDevice() = 0;
  virtual ~DeviceFactory();

protected:
  /*
   * This helper method returns the extension of the provided fileName. The
   * fileName is expected to be ASCII encoded and have an extension which is 3
   * letters long
   */
  std::string extractExtension(const std::string& fileName);
};

} /* namespace mtcapy */

#endif /* DEVICEFACTORY_H_ */
