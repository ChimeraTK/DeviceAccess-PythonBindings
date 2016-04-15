#ifndef REGISTERACCESSORWRAPPERFUNCTIONS_H_
#define REGISTERACCESSORWRAPPERFUNCTIONS_H_

#include <mtca4u/Device.h>
#include <mtca4u/TwoDRegisterAccessor.h>
#include "HelperFunctions.h"

namespace mtca4upy {

/*
 * Returns mtca4u-deviceaccess type Device. Inputs are the device identifier
 * (/dev/..) and the location of the map file. A mtca4u-deviceaccess Dummy
 * device is returned if both the  deviceIdentifier and mapFile parameters are
 * set to the same valid Map file.
 */
boost::shared_ptr<mtca4u::Device> createDevice(
    const std::string &deviceIdentifier, const std::string &mapFile);

/*
 * This method uses the factory provided by the device access library for device
 * creation. The deviceAlias is picked from the specified dmap file, which is
 * set through the environment variable DMAP_PATH_ENV
 */

boost::shared_ptr<mtca4u::Device> createDevice(const std::string &deviceAlias);

namespace RegisterAccessor {

  // FIXME: extract 'size_t arraySize' from  bp::numeric array.
  void read(mtca4u::Device::RegisterAccessor &self,
            bp::numeric::array &numpyArray, size_t numberOfElementsToRead,
            uint32_t startIndexInRegister);

  void readRaw(mtca4u::Device::RegisterAccessor &self,
               bp::numeric::array &numpyArray, size_t numberOfElementsToRead,
               uint32_t startIndexInRegister);

  void readDMARaw(mtca4u::Device::RegisterAccessor &self,
                  bp::numeric::array &numpyArray, size_t numberOfElementsToRead,
                  uint32_t startIndexInRegister);

  void write(mtca4u::Device::RegisterAccessor &self,
             bp::numeric::array &numpyArray, size_t numElementsToWrite,
             uint32_t startIndexInRegister);

  void writeRaw(mtca4u::Device::RegisterAccessor &self,
                bp::numeric::array &numpyArray, size_t numElementsToWrite,
                uint32_t startIndexInRegister);

  uint32_t size(mtca4u::Device::RegisterAccessor &self);
}

namespace MuxDataAccessor {

  void readInDataFromCard(mtca4u::TwoDRegisterAccessor<float> &self);

  size_t getSequenceCount(mtca4u::TwoDRegisterAccessor<float> &self);

  size_t getBlockCount(mtca4u::TwoDRegisterAccessor<float> &self);

  void copyReadInData(mtca4u::TwoDRegisterAccessor<float> &self,
                      bp::numeric::array &numpyArray);

} // namespace mtca4upy::MuxDataAccessor

namespace DeviceAccess {
  /**
   * return register Accessor from mtca4u-deviceacces Device
   */
  boost::shared_ptr<mtca4u::Device::RegisterAccessor> getRegisterAccessor(
      const mtca4u::Device &self, const std::string &moduleName,
      const std::string &regName);

  mtca4u::TwoDRegisterAccessor<float>
  getMultiplexedDataAccessor(const mtca4u::Device &self,
                             const std::string &moduleName,
                             const std::string &regionName);

  void writeRaw(mtca4u::Device &self, uint32_t regOffset,
                bp::numeric::array dataToWrite, size_t bytesToWrite,
                uint8_t bar);
} // namespace mtca4upy::deviceAccess

void setDmapFile(const std::string &dmapFile);
std::string getDmapFile();

} // namespace mtca4upy

#endif /* REGISTERACCESSORWRAPPERFUNCTIONS_H_ */
