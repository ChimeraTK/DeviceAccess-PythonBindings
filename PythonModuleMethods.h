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

namespace OneDAccessor {
  template <typename T>
  void read(mtca4u::OneDRegisterAccessor<T> &self,
            mtca4upy::NumpyObject &numpyArray){
	  self.read();
      T* allocatedSpace =
          reinterpret_cast<T*>(extractDataPointer(numpyArray));
      for(const auto &element: self){
    	  *(allocatedSpace++) = element;
      }
  }

  template <typename T>
  void write(mtca4u::OneDRegisterAccessor<T> &self,
             mtca4upy::NumpyObject &numpyArray) {
    T *dataToWrite = reinterpret_cast<T *>(extractDataPointer(numpyArray));
    unsigned int numberOfElementsToWrite = self.getNElements();
    for (size_t index = 0; index < numberOfElementsToWrite; ++index) {
      self[index] = dataToWrite[index];
    }
    self.write();
  }

  template <typename T>
  size_t getNumberOfElements(mtca4u::OneDRegisterAccessor<T> &self){
	  return self.getNElements();
  }
}

namespace TwoDAccessor {
template <typename T>
void read(mtca4u::TwoDRegisterAccessor<T> &self,
          mtca4upy::NumpyObject &numpyArray){

	  self.read();

    T* allocatedSpace =
        reinterpret_cast<T*>(extractDataPointer(numpyArray));

    auto numSequences = self.getNChannels();
    auto elemetsInEachSequence = self.getNElementsPerChannel();

    // pyArrayCol corresponds to the sequence numbers and pyArrrayRow to
    // each element of the sequence
    for (size_t pyArrayCol = 0; pyArrayCol < numSequences; ++pyArrayCol) {
      for (size_t pyArrrayRow = 0; pyArrrayRow < elemetsInEachSequence; ++pyArrrayRow) {
    	  allocatedSpace[(numSequences * pyArrrayRow) + pyArrayCol] =
            self[pyArrayCol][pyArrrayRow];
      }
    }
}

template <typename T>
size_t getNChannels(mtca4u::TwoDRegisterAccessor<T> &self){
	  return self.getNChannels();
}

template <typename T>
size_t getNElementsPerChannel(mtca4u::TwoDRegisterAccessor<T> &self){
	  return self.getNElementsPerChannel();
}

} //namespace TwoDAccessor


namespace DeviceAccess {
  mtca4u::TwoDRegisterAccessor<float> getTwoDAccessor(
      const mtca4u::Device &self, const std::string &registerPath);

  template <typename T>
  mtca4u::OneDRegisterAccessor<T> getOneDAccessor(
      const mtca4u::Device& self, const std::string& registerPath,
      size_t numberOfelementsToRead, size_t elementOffset) {
    return self.getOneDRegisterAccessor<T>(registerPath, numberOfelementsToRead,
                                           elementOffset);
  }

  mtca4u::OneDRegisterAccessor<int32_t> getRawOneDAccessor(
      const mtca4u::Device &self, const std::string &registerPath,
      size_t numberOfelementsToRead, size_t elementOffset);

  void writeRaw(mtca4u::Device &self, uint32_t regOffset,
                mtca4upy::NumpyObject dataToWrite, size_t bytesToWrite,
                uint8_t bar);

} // namespace mtca4upy::deviceAccess

void setDmapFile(const std::string &dmapFile);
std::string getDmapFile();

} // namespace mtca4upy

#endif /* REGISTERACCESSORWRAPPERFUNCTIONS_H_ */
