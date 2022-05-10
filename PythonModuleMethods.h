#ifndef REGISTERACCESSORWRAPPERFUNCTIONS_H_
#define REGISTERACCESSORWRAPPERFUNCTIONS_H_

//#include "HelperFunctions.h"
#include <ChimeraTK/Device.h>
#include <ChimeraTK/TwoDRegisterAccessor.h>
#include <boost/python/numpy.hpp>
#include <vector>

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
    /*
      template<typename T>
    void read(ChimeraTK::TwoDRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      /*
      std::vector<std::vector<T>> buffer = {{1, 2, 3}, {4, 5, 9}};

      np::ndarray np_buffer = np::from_data(buffer.data(),           // data ->
          np::dtype::get_builtin<T>(),                               // dtype -> T
          p::make_tuple(channels, elementsPerChannel),               // shape -> size
          p::make_tuple(sizeof(T) * 0, sizeof(T) * 1), p::object()); // stride = 1*1

      std::vector<std::vector<int32_t>> originalVector = {{5, 5, 3, 5, 905}, {23, 15, 33, 35, 5}};

      // std::cout << "sizeOfArray = " << sizeOfVector << std::endl;

      np::ndarray py_array = np::from_data(

          originalVector.data(), // origin

          np::dtype::get_builtin<T>(), // type

          p::make_tuple(2, 5), // shape

          p::make_tuple(0, 1), // stride

          p::object() // owner

      );

      np::ndarray copy_array = np::empty(p::make_tuple(elementsPerChannel, channels), np::dtype::get_builtin<T>());

      for(int i = 0; i < channels; ++i) {
        for(int j = 0; j < elementsPerChannel; ++j) {
          copy_array[i][j] = self[i][j];
        }
      }

      return copy_array; // .copy(); // https://stackoverflow.com/a/54888186/4919081
      
      self.read();
      copyUserBufferToNumpyNDArray(self, np_buffer);
    }

    template<typename T>
    bool readLatest(ChimeraTK::TwoDRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      bool hasNewData = self.readLatest();
      if(hasNewData) copyUserBufferToNumpyNDArray(self, np_buffer);
      return hasNewData;
    }

    template<typename T>
    bool readNonBlocking(ChimeraTK::TwoDRegisterAccessor<T>& self, np::ndarray& np_buffer) {
      bool hasNewData = self.readNonBlocking();
      if(hasNewData) copyUserBufferToNumpyNDArray(self, np_buffer);
      return hasNewData;
    }
    */
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

    template<typename T>
    const std::string getName(ChimeraTK::TwoDRegisterAccessor<T>& self) {
      return self.getName();
    }

    template<typename T>
    const std::string getUnit(ChimeraTK::TwoDRegisterAccessor<T>& self) {
      return self.getUnit();
    }

    template<typename T>
    const std::string getDescription(ChimeraTK::TwoDRegisterAccessor<T>& self) {
      return self.getDescription();
    }

    template<typename T>
    bool isReadOnly(ChimeraTK::TwoDRegisterAccessor<T>& self) {
      return self.isReadOnly();
    }

    template<typename T>
    bool isReadable(ChimeraTK::TwoDRegisterAccessor<T>& self) {
      return self.isReadable();
    }

    template<typename T>
    bool isWriteable(ChimeraTK::TwoDRegisterAccessor<T>& self) {
      return self.isWriteable();
    }

    template<typename T>
    bool isInitialised(ChimeraTK::TwoDRegisterAccessor<T>& self) {
      return self.isInitialised();
    }

    template<typename T>
    void setDataValidity(ChimeraTK::TwoDRegisterAccessor<T>& self, ChimeraTK::DataValidity valid) {
      self.setDataValidity(valid);
    }

    template<typename T>
    ChimeraTK::DataValidity dataValidity(ChimeraTK::TwoDRegisterAccessor<T>& self) {
      return self.dataValidity();
    }
  } // namespace TwoDRegisterAccessor

  namespace DeviceAccess {
    ChimeraTK::TwoDRegisterAccessor<double> getTwoDAccessor(
        const ChimeraTK::Device& self, const std::string& registerPath);

    template<typename T>
    ChimeraTK::TwoDRegisterAccessor<T> getGeneralTwoDAccessor(const ChimeraTK::Device& self,
        const std::string& registerPath, size_t numberOfElements, size_t elementsOffset, boost::python::list flaglist) {
      ChimeraTK::AccessModeFlags flags{};
      size_t count = len((flaglist));
      for(size_t i = 0; i < count; i++) {
        flags.add(p::extract<ChimeraTK::AccessMode>(flaglist.pop()));
      }

      return self.getTwoDRegisterAccessor<T>(registerPath, numberOfElements, elementsOffset, flags);
    }

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

  } // namespace DeviceAccess

  void setDmapFile(const std::string& dmapFile);
  std::string getDmapFile();

} // namespace mtca4upy

#endif /* REGISTERACCESSORWRAPPERFUNCTIONS_H_ */
