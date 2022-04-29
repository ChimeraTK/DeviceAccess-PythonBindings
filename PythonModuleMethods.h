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
    np::ndarray getBuffer(ChimeraTK::TwoDRegisterAccessor<T>& self) {
      size_t channels = self.getNChannels();
      size_t elementsPerChannel = self.getNElementsPerChannel();
      /*
      std::vector<std::vector<T>> buffer = {{1, 2, 3}, {4, 5, 9}};

      np::ndarray np_buffer = np::from_data(buffer.data(), // data ->
          np::dtype::get_builtin<T>(),                     // dtype -> double
          p::make_tuple(buffer.size()),                    // shape -> size
          p::make_tuple(sizeof(T)), p::object());          // stride = 1

      */

      return np_buffer;
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

    std::string getCatalogueMetadata(const ChimeraTK::Device& self, const std::string& parameterName);

    void open(ChimeraTK::Device& self, std::string const& aliasName);
    void open(ChimeraTK::Device& self);
    void close(ChimeraTK::Device& self);

  } // namespace DeviceAccess

  void setDmapFile(const std::string& dmapFile);
  std::string getDmapFile();

} // namespace mtca4upy

#endif /* REGISTERACCESSORWRAPPERFUNCTIONS_H_ */
