#ifndef REGISTERACCESSORWRAPPERFUNCTIONS_H_
#define REGISTERACCESSORWRAPPERFUNCTIONS_H_

//#include "HelperFunctions.h"
#include <ChimeraTK/Device.h>
#include <ChimeraTK/OneDRegisterAccessor.h>
#include <ChimeraTK/RegisterCatalogue.h>
#include <ChimeraTK/RegisterInfo.h>
#include <ChimeraTK/TransferElementID.h>
#include <ChimeraTK/TwoDRegisterAccessor.h>
#include <ChimeraTK/VoidRegisterAccessor.h>

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
  boost::shared_ptr<ChimeraTK::Device> getDevice_no_alias();
  boost::shared_ptr<ChimeraTK::Device> getDevice(const std::string& deviceAlias);

  namespace GeneralRegisterAccessor {
    template<typename T>
    const std::string getName(T& self) {
      return self.getName();
    }

    template<typename T>
    const std::string getUnit(T& self) {
      return self.getUnit();
    }

    template<typename T>
    const std::string getDescription(T& self) {
      return self.getDescription();
    }

    template<typename T>
    bool isReadOnly(T& self) {
      return self.isReadOnly();
    }

    template<typename T>
    bool isReadable(T& self) {
      return self.isReadable();
    }

    template<typename T>
    bool isWriteable(T& self) {
      return self.isWriteable();
    }

    template<typename T>
    bool isInitialised(T& self) {
      return self.isInitialised();
    }

    template<typename T>
    void setDataValidity(T& self, ChimeraTK::DataValidity valid) {
      self.setDataValidity(valid);
    }

    template<typename T>
    ChimeraTK::DataValidity dataValidity(T& self) {
      return self.dataValidity();
    }
    template<typename T>
    ChimeraTK::TransferElementID getId(T& self) {
      return self.getId();
    }

    template<typename T>
    ChimeraTK::VersionNumber getVersionNumber(T& self) {
      return self.getVersionNumber();
    }
  } // namespace GeneralRegisterAccessor

  namespace VoidRegisterAccessor {

    bool write(ChimeraTK::VoidRegisterAccessor& self);

    void read(ChimeraTK::VoidRegisterAccessor& self);

    bool readNonBlocking(ChimeraTK::VoidRegisterAccessor& self);

    bool readLatest(ChimeraTK::VoidRegisterAccessor& self);

  } // namespace VoidRegisterAccessor

  namespace ScalarRegisterAccessor {

    template<typename T>
    void copyNpArrayToUserBuffer(ChimeraTK::ScalarRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      T* input_ptr = reinterpret_cast<T*>(np_buffer.get_data());
      self = *(input_ptr);
    }

    template<typename T>
    void copyUserBufferToNpArray(ChimeraTK::ScalarRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      np_buffer[0] = static_cast<T>(self);
    }

    template<>
    void copyUserBufferToNpArray<ChimeraTK::Boolean>(
        ChimeraTK::ScalarRegisterAccessor<ChimeraTK::Boolean>& self, np::ndarray& np_buffer);

    template<typename T>
    bool write(ChimeraTK::ScalarRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      copyNpArrayToUserBuffer(self, np_buffer);
      return self.write();
    }

    template<typename T>
    bool writeDestructively(ChimeraTK::ScalarRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      copyNpArrayToUserBuffer(self, np_buffer);
      return self.writeDestructively();
    }

    template<typename T>
    void read(ChimeraTK::ScalarRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      self.read();
      copyUserBufferToNpArray(self, np_buffer);
    }

    template<typename T>
    bool readNonBlocking(ChimeraTK::ScalarRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      bool status = self.readNonBlocking();
      copyUserBufferToNpArray(self, np_buffer);
      return status;
    }

    template<typename T>
    bool readLatest(ChimeraTK::ScalarRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      bool status = self.readLatest();
      copyUserBufferToNpArray(self, np_buffer);
      return status;
    }

    template<typename T>
    T readAndGet(ChimeraTK::ScalarRegisterAccessor<T>& self) {
      return self.readAndGet();
    }

    template<typename T>
    void setAndWrite(ChimeraTK::ScalarRegisterAccessor<T>& self, T newValue, ChimeraTK::VersionNumber versionNumber) {
      self.setAndWrite(newValue, versionNumber);
    }

    template<typename T>
    void writeIfDifferent(
        ChimeraTK::ScalarRegisterAccessor<T>& self, T newValue, ChimeraTK::VersionNumber versionNumber) {
      self.writeIfDifferent(newValue, versionNumber);
    }

  } // namespace ScalarRegisterAccessor

  namespace OneDRegisterAccessor {

    // supposed to skip the copy mechanism of the user buffer to the np-array, but
    // is gliched in the current version.
    template<typename T>
    void linkUserBufferToNpArray(ChimeraTK::OneDRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      np_buffer = np::from_data(self.data(),  // data ->
          np::dtype::get_builtin<T>(),        // dtype -> T
          p::make_tuple(self.getNElements()), // shape -> size
          p::make_tuple(sizeof(T)),           // stride = 1*1
          p::object());                       // owner
    }

    template<typename T>
    void copyUserBufferToNpArray(ChimeraTK::OneDRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      size_t elements = self.getNElements();
      for(size_t i = 0; i < elements; ++i) {
        np_buffer[i] = self[i];
      }
    }

    template<typename T>
    void copyNpArrayToUserBuffer(ChimeraTK::OneDRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      size_t elements = self.getNElements();
      T* input_ptr = reinterpret_cast<T*>(np_buffer.get_data());
      for(size_t i = 0; i < elements; ++i) {
        self[i] = *(input_ptr + i);
      }
    }

    template<typename T>
    bool write(ChimeraTK::OneDRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      copyNpArrayToUserBuffer(self, np_buffer);
      return self.write();
    }

    template<typename T>
    bool writeDestructively(ChimeraTK::OneDRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      copyNpArrayToUserBuffer(self, np_buffer);
      return self.writeDestructively();
    }

    template<typename T>
    int getNElements(ChimeraTK::OneDRegisterAccessor<T>& self) {
      return self.getNElements();
    }

    template<typename T>
    void read(ChimeraTK::OneDRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      self.read();
      copyUserBufferToNpArray(self, np_buffer);
    }

    template<typename T>
    bool readNonBlocking(ChimeraTK::OneDRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      bool status = self.readNonBlocking();
      copyUserBufferToNpArray(self, np_buffer);
      return status;
    }

    template<typename T>
    bool readLatest(ChimeraTK::OneDRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      bool status = self.readLatest();
      copyUserBufferToNpArray(self, np_buffer);
      return status;
    }
  } // namespace OneDRegisterAccessor

  namespace TwoDRegisterAccessor {

    template<typename T>
    void copyUserBufferToNumpyNDArray(ChimeraTK::TwoDRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      size_t channels = self.getNChannels();
      size_t elementsPerChannel = self.getNElementsPerChannel();
      for(size_t i = 0; i < channels; ++i) {
        for(size_t j = 0; j < elementsPerChannel; ++j) {
          np_buffer[i][j] = self[i][j];
        }
      }
    }

    template<typename T, typename ReadFunction>
    bool genericReadFuntion(
        ChimeraTK::TwoDRegisterAccessor<T>& self, np::ndarray& np_buffer, ReadFunction readFunction) {
      bool hasNewData = readFunction();
      if(hasNewData) copyUserBufferToNumpyNDArray(self, np_buffer);
      return hasNewData;
    }
    template<typename T>
    void read(ChimeraTK::TwoDRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      genericReadFuntion(self, np_buffer, [&]() {
        self.read();
        return true;
      });
    }

    template<typename T>
    bool readNonBlocking(ChimeraTK::TwoDRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      return genericReadFuntion(self, np_buffer, [&]() { return self.readNonBlocking(); });
    }

    template<typename T>
    bool readLatest(ChimeraTK::TwoDRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      return genericReadFuntion(self, np_buffer, [&]() { return self.readLatest(); });
    }

    template<typename T>
    void transferNumpyArrayToUserBuffer(ChimeraTK::TwoDRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      size_t channels = self.getNChannels();
      size_t elementsPerChannel = self.getNElementsPerChannel();
      T* input_ptr = reinterpret_cast<T*>(np_buffer.get_data());
      for(size_t i = 0; i < channels; ++i) {
        for(size_t j = 0; j < elementsPerChannel; ++j) {
          self[i][j] = *(input_ptr + j + (i * elementsPerChannel));
        }
      }
    }

    template<typename T>
    bool write(ChimeraTK::TwoDRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      transferNumpyArrayToUserBuffer(self, np_buffer);
      return self.write();
    }

    template<typename T>
    bool writeDestructively(ChimeraTK::TwoDRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      transferNumpyArrayToUserBuffer(self, np_buffer);
      return self.writeDestructively();
    }

    template<typename T>
    int getNChannels(ChimeraTK::TwoDRegisterAccessor<T>& self) {
      return self.getNChannels();
    }
    template<typename T>
    int getNElementsPerChannel(ChimeraTK::TwoDRegisterAccessor<T>& self) {
      return self.getNElementsPerChannel();
    }

  } // namespace TwoDRegisterAccessor

  namespace DeviceAccess {
    ChimeraTK::TwoDRegisterAccessor<double> getTwoDAccessor(
        const ChimeraTK::Device& self, const std::string& registerPath);

    ChimeraTK::AccessModeFlags convertFlagsFromPython(boost::python::list flaglist);

    template<typename T>
    ChimeraTK::TwoDRegisterAccessor<T> getGeneralTwoDAccessor(const ChimeraTK::Device& self,
        const std::string& registerPath, size_t numberOfElements, size_t elementsOffset, boost::python::list flaglist) {
      return self.getTwoDRegisterAccessor<T>(
          registerPath, numberOfElements, elementsOffset, convertFlagsFromPython(flaglist));
    }

    template<typename T>
    ChimeraTK::OneDRegisterAccessor<T> getGeneralOneDAccessor(const ChimeraTK::Device& self,
        const std::string& registerPath, size_t numberOfElements, size_t elementsOffset, boost::python::list flaglist) {
      return self.getOneDRegisterAccessor<T>(
          registerPath, numberOfElements, elementsOffset, convertFlagsFromPython(flaglist));
    }

    template<typename T>
    ChimeraTK::ScalarRegisterAccessor<T> getGeneralScalarAccessor(const ChimeraTK::Device& self,
        const std::string& registerPath, size_t elementsOffset, boost::python::list flaglist) {
      return self.getScalarRegisterAccessor<T>(registerPath, elementsOffset, convertFlagsFromPython(flaglist));
    }

    ChimeraTK::VoidRegisterAccessor getVoidRegisterAccessor(
        const ChimeraTK::Device& self, const std::string& registerPath, boost::python::list flaglist);

    template<typename T>
    ChimeraTK::OneDRegisterAccessor<T> getOneDAccessor(const ChimeraTK::Device& self, const std::string& registerPath,
        size_t numberOfelementsToRead, size_t elementOffset) {
      return self.getOneDRegisterAccessor<T>(registerPath, numberOfelementsToRead, elementOffset);
    }

    ChimeraTK::OneDRegisterAccessor<int32_t> getRawOneDAccessor(const ChimeraTK::Device& self,
        const std::string& registerPath, size_t numberOfelementsToRead, size_t elementOffset);

    std::string getCatalogueMetadata(const ChimeraTK::Device& self, const std::string& parameterName);

    void open(ChimeraTK::Device& self, std::string const& aliasName);
    void open(ChimeraTK::Device& self);
    void close(ChimeraTK::Device& self);

    void activateAsyncRead(ChimeraTK::Device& self);
    ChimeraTK::RegisterCatalogue getRegisterCatalogue(ChimeraTK::Device& self);

    void write(const ChimeraTK::Device& self, np::ndarray& arr, const std::string& registerPath,
        size_t numberOfElements, size_t elementsOffset, boost::python::list flaglist);
    np::ndarray read(const ChimeraTK::Device& self, np::ndarray& arr, const std::string& registerPath,
        size_t numberOfElements, size_t elementsOffset, boost::python::list flaglist);
    ChimeraTK::DataType convert_dytpe_to_usertype(np::dtype dtype);

  } // namespace DeviceAccess

  void setDmapFile(const std::string& dmapFile);
  std::string getDmapFile();

  namespace TransferElementID {
    bool isValid(ChimeraTK::TransferElementID& self);
    bool lt(ChimeraTK::TransferElementID& self, ChimeraTK::TransferElementID& other);
    bool le(ChimeraTK::TransferElementID& self, ChimeraTK::TransferElementID& other);
    bool eq(ChimeraTK::TransferElementID& self, ChimeraTK::TransferElementID& other);
    bool gt(ChimeraTK::TransferElementID& self, ChimeraTK::TransferElementID& other);
    bool ge(ChimeraTK::TransferElementID& self, ChimeraTK::TransferElementID& other);
    bool ne(ChimeraTK::TransferElementID& self, ChimeraTK::TransferElementID& other);
  } // namespace TransferElementID

  namespace VersionNumber {
    std::string str(ChimeraTK::VersionNumber& self);
    boost::posix_time::ptime getTime(ChimeraTK::VersionNumber& self);
    bool lt(ChimeraTK::VersionNumber& self, ChimeraTK::VersionNumber& other);
    bool le(ChimeraTK::VersionNumber& self, ChimeraTK::VersionNumber& other);
    bool eq(ChimeraTK::VersionNumber& self, ChimeraTK::VersionNumber& other);
    bool gt(ChimeraTK::VersionNumber& self, ChimeraTK::VersionNumber& other);
    bool ge(ChimeraTK::VersionNumber& self, ChimeraTK::VersionNumber& other);
    bool ne(ChimeraTK::VersionNumber& self, ChimeraTK::VersionNumber& other);
    ChimeraTK::VersionNumber getNullVersion();
  } // namespace VersionNumber

  namespace RegisterCatalogue {
    bool hasRegister(ChimeraTK::RegisterCatalogue& self, const std::string& registerPathName);
    ChimeraTK::RegisterInfo getRegister(ChimeraTK::RegisterCatalogue& self, const std::string& registerPathName);
  } // namespace RegisterCatalogue

  namespace RegisterInfo {
    unsigned int getNumberOfElements(ChimeraTK::RegisterInfo& self);
    unsigned int getNumberOfChannels(ChimeraTK::RegisterInfo& self);
    unsigned int getNumberOfDimensions(ChimeraTK::RegisterInfo& self);
    bool isReadable(ChimeraTK::RegisterInfo& self);
    bool isValid(ChimeraTK::RegisterInfo& self);
    bool isWriteable(ChimeraTK::RegisterInfo& self);
    std::string getRegisterName(ChimeraTK::RegisterInfo& self);
    boost::python::list getSupportedAccessModes(ChimeraTK::RegisterInfo& self);
  } // namespace RegisterInfo

} // namespace mtca4upy

#endif /* REGISTERACCESSORWRAPPERFUNCTIONS_H_ */
