#ifndef REGISTERACCESSORWRAPPERFUNCTIONS_H_
#define REGISTERACCESSORWRAPPERFUNCTIONS_H_

#include "HelperFunctions.h"
#include <ChimeraTK/Device.h>
#include <ChimeraTK/TwoDRegisterAccessor.h>
#include <boost/python/numpy.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;

namespace mtca4upy {

  /*
   * Returns mtca4u-deviceaccess type Device. Inputs are the device identifier
   * (/dev/..) and the location of the map file. A mtca4u-deviceaccess Dummy
   * device is returned if both the  deviceIdentifier and mapFile parameters are
   * set to the same valid Map file.
   */
  boost::shared_ptr<ChimeraTK::Device> createDevice(const std::string& deviceIdentifier, const std::string& mapFile);

  /*
   * This method uses the factory provided by the device access library for device
   * creation. The deviceAlias is picked from the specified dmap file, which is
   * set through the environment variable DMAP_PATH_ENV
   */

  boost::shared_ptr<ChimeraTK::Device> createDevice(const std::string& deviceAlias);
  boost::shared_ptr<ChimeraTK::Device> getDevice();
  boost::shared_ptr<ChimeraTK::Device> getDevice(const std::string& deviceAlias);

  namespace OneDAccessor {
    template<typename T>
    void read(ChimeraTK::OneDRegisterAccessor<T>& self, mtca4upy::NumpyObject& numpyArray) {
      self.read();
      T* allocatedSpace = reinterpret_cast<T*>(extractDataPointer(numpyArray));
      for(const auto& element : self) {
        *(allocatedSpace++) = element;
      }
    }

    template<typename T>
    void write(ChimeraTK::OneDRegisterAccessor<T>& self, mtca4upy::NumpyObject& numpyArray) {
      T* dataToWrite = reinterpret_cast<T*>(extractDataPointer(numpyArray));
      unsigned int numberOfElementsToWrite = self.getNElements();
      for(size_t index = 0; index < numberOfElementsToWrite; ++index) {
        self[index] = dataToWrite[index];
      }
      self.write();
    }

    template<typename T>
    size_t getNumberOfElements(ChimeraTK::OneDRegisterAccessor<T>& self) {
      return self.getNElements();
    }
  } // namespace OneDAccessor

  namespace TwoDAccessor {
    template<typename T>
    void read(ChimeraTK::TwoDRegisterAccessor<T>& self, mtca4upy::NumpyObject& numpyArray) {
      self.read();

      T* allocatedSpace = reinterpret_cast<T*>(extractDataPointer(numpyArray));

      auto numSequences = self.getNChannels();
      auto elemetsInEachSequence = self.getNElementsPerChannel();

      // pyArrayCol corresponds to the sequence numbers and pyArrrayRow to
      // each element of the sequence
      for(size_t pyArrayCol = 0; pyArrayCol < numSequences; ++pyArrayCol) {
        for(size_t pyArrrayRow = 0; pyArrrayRow < elemetsInEachSequence; ++pyArrrayRow) {
          allocatedSpace[(numSequences * pyArrrayRow) + pyArrayCol] = self[pyArrayCol][pyArrrayRow];
        }
      }
    }

    template<typename T>
    size_t getNChannels(ChimeraTK::TwoDRegisterAccessor<T>& self) {
      return self.getNChannels();
    }

    template<typename T>
    size_t getNElementsPerChannel(ChimeraTK::TwoDRegisterAccessor<T>& self) {
      return self.getNElementsPerChannel();
    }

  } // namespace TwoDAccessor

  namespace TwoDRegisterAccessor {
    template<typename T>
    std::vector<std::vector<T>> getBuffer(ChimeraTK::TwoDRegisterAccessor<T>& self) {
      size_t channels = self.getNChannels();
      size_t elementsPerChannel = self.getNElementsPerChannel();
      return {{1, 2, 3}, {4, 5, 6}};
    }
    /*
    template<typename T>
    void getBuffer(ChimeraTK::TwoDRegisterAccessor<T>& self) {
      return;
    }
  */
  } // namespace TwoDRegisterAccessor

  namespace DeviceAccess {
    ChimeraTK::TwoDRegisterAccessor<double> getTwoDAccessor(
        const ChimeraTK::Device& self, const std::string& registerPath);

    template<typename T>
    ChimeraTK::TwoDRegisterAccessor<T> getGeneralTwoDAccessor(
        const ChimeraTK::Device& self, const std::string& registerPath, size_t numberOfElements, size_t elementsOffset
        //const ChimeraTK::AccessModeFlags& flags
    ) {
      return self.getTwoDRegisterAccessor<T>(registerPath, numberOfElements, elementsOffset, {});
      // flags);
    }

    template<typename T>
    ChimeraTK::OneDRegisterAccessor<T> getOneDAccessor(const ChimeraTK::Device& self, const std::string& registerPath,
        size_t numberOfelementsToRead, size_t elementOffset) {
      return self.getOneDRegisterAccessor<T>(registerPath, numberOfelementsToRead, elementOffset);
    }

    ChimeraTK::OneDRegisterAccessor<int32_t> getRawOneDAccessor(const ChimeraTK::Device& self,
        const std::string& registerPath, size_t numberOfelementsToRead, size_t elementOffset);

    void writeRaw(ChimeraTK::Device& self, std::string const& registerName, //
        uint32_t regOffset, mtca4upy::NumpyObject dataToWrite, size_t bytesToWrite);

    std::string getCatalogueMetadata(const ChimeraTK::Device& self, const std::string& parameterName);

    void open(ChimeraTK::Device& self, std::string const& aliasName);
    void open(ChimeraTK::Device& self);
    void close(ChimeraTK::Device& self);

  } // namespace DeviceAccess

  void setDmapFile(const std::string& dmapFile);
  std::string getDmapFile();

} // namespace mtca4upy

#endif /* REGISTERACCESSORWRAPPERFUNCTIONS_H_ */
