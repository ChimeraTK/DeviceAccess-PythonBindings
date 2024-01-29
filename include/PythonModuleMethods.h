// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <ChimeraTK/Device.h>
#include <ChimeraTK/OneDRegisterAccessor.h>
#include <ChimeraTK/RegisterCatalogue.h>
#include <ChimeraTK/RegisterInfo.h>
#include <ChimeraTK/TransferElementID.h>
#include <ChimeraTK/TwoDRegisterAccessor.h>
#include <ChimeraTK/VoidRegisterAccessor.h>

#include <boost/python/numpy.hpp>

#include <codecvt>
#include <locale>

namespace p = boost::python;
namespace np = boost::python::numpy;
namespace ctk = ChimeraTK;

/*****************************************************************************************************************/

namespace boost::python::numpy::detail {

  /**
   * Provide numpy dtype for ctk::Boolean (conversion is identical as for plain bool)
   */
  template<>
  struct builtin_dtype<ctk::Boolean, false> {
    static dtype get() { return builtin_dtype<bool, true>::get(); }
  };

  /**
   * Provide numpy dtype for std::string (conversion is identical as for plain bool)
   */
  template<>
  struct builtin_dtype<std::string, false> {
    static dtype get() { return builtin_dtype<char, true>::get(); }
  };

} // namespace boost::python::numpy::detail

/**
 * Provide converter from ctk::Boolean into Python bool type
 */
struct CtkBoolean_to_python {
  static PyObject* convert(ctk::Boolean const& value) {
    return boost::python::incref(boost::python::object(bool(value)).ptr());
  }
};

/*****************************************************************************************************************/
/*****************************************************************************************************************/

namespace mtca4upy {

  /*
   * Returns mtca4u-deviceaccess type Device. Inputs are the device identifier
   * (/dev/..) and the location of the map file. A mtca4u-deviceaccess Dummy
   * device is returned if both the  deviceIdentifier and mapFile parameters are
   * set to the same valid Map file.
   */
  boost::shared_ptr<ctk::Device> createDevice(const std::string& deviceIdentifier, const std::string& mapFile);

  /*
   * This method uses the factory provided by the device access library for device
   * creation. The deviceAlias is picked from the specified dmap file, which is
   * set through the environment variable DMAP_PATH_ENV
   */

  boost::shared_ptr<ctk::Device> createDevice(const std::string& deviceAlias);
  boost::shared_ptr<ctk::Device> getDevice_no_alias();
  boost::shared_ptr<ctk::Device> getDevice(const std::string& deviceAlias);

  void setDmapFile(const std::string& dmapFile);
  std::string getDmapFile();

  /*****************************************************************************************************************/
  /*****************************************************************************************************************/

  /** (Static) class to map ctk::Device to python */
  class Device {
   public:
    static ctk::TwoDRegisterAccessor<double> getTwoDAccessor(const ctk::Device& self, const std::string& registerPath);

    static ctk::AccessModeFlags convertFlagsFromPython(boost::python::list flaglist);

    template<typename T>
    static ctk::TwoDRegisterAccessor<T> getGeneralTwoDAccessor(const ctk::Device& self, const std::string& registerPath,
        size_t numberOfElements, size_t elementsOffset, boost::python::list flaglist) {
      return self.getTwoDRegisterAccessor<T>(
          registerPath, numberOfElements, elementsOffset, convertFlagsFromPython(flaglist));
    }

    template<typename T>
    static ctk::OneDRegisterAccessor<T> getGeneralOneDAccessor(const ctk::Device& self, const std::string& registerPath,
        size_t numberOfElements, size_t elementsOffset, boost::python::list flaglist) {
      return self.getOneDRegisterAccessor<T>(
          registerPath, numberOfElements, elementsOffset, convertFlagsFromPython(flaglist));
    }

    template<typename T>
    static ctk::ScalarRegisterAccessor<T> getGeneralScalarAccessor(
        const ctk::Device& self, const std::string& registerPath, size_t elementsOffset, boost::python::list flaglist) {
      return self.getScalarRegisterAccessor<T>(registerPath, elementsOffset, convertFlagsFromPython(flaglist));
    }

    static ctk::VoidRegisterAccessor getVoidRegisterAccessor(
        const ctk::Device& self, const std::string& registerPath, boost::python::list flaglist);

    template<typename T>
    static ctk::OneDRegisterAccessor<T> getOneDAccessor(
        const ctk::Device& self, const std::string& registerPath, size_t numberOfelementsToRead, size_t elementOffset) {
      return self.getOneDRegisterAccessor<T>(registerPath, numberOfelementsToRead, elementOffset);
    }

    static ctk::OneDRegisterAccessor<int32_t> getRawOneDAccessor(
        const ctk::Device& self, const std::string& registerPath, size_t numberOfelementsToRead, size_t elementOffset);

    static std::string getCatalogueMetadata(const ctk::Device& self, const std::string& parameterName);

    static void open(ctk::Device& self, std::string const& aliasName);
    static void open(ctk::Device& self);
    static void close(ctk::Device& self);

    static void activateAsyncRead(ctk::Device& self);
    static ctk::RegisterCatalogue getRegisterCatalogue(ctk::Device& self);

    static void write(const ctk::Device& self, np::ndarray& arr, const std::string& registerPath,
        size_t numberOfElements, size_t elementsOffset, boost::python::list flaglist);

    static np::ndarray read(const ctk::Device& self, const std::string& registerPath, size_t numberOfElements,
        size_t elementsOffset, boost::python::list flaglist);
  };

  /*****************************************************************************************************************/
  /*****************************************************************************************************************/

  /**
   * (Static) class implementing register accessor functions for all accessor types
   */
  template<typename ACCESSOR>
  class GeneralRegisterAccessor {
   public:
    // "convert" the return type from a const string reference to a real string, since boost python cannot deal with
    // that otherwise
    static std::string getName(ACCESSOR& self) { return self.getName(); }

    static std::string getUnit(ACCESSOR& self) { return self.getUnit(); }

    static std::string getDescription(ACCESSOR& self) { return self.getDescription(); }

    static std::string getAccessModeFlagsString(ACCESSOR& self) { return self.getAccessModeFlags().serialize(); }

    static np::ndarray read(ACCESSOR& self, np::ndarray& np_buffer);

    static auto readNonBlocking(ACCESSOR& self, np::ndarray& np_buffer);

    static auto readLatest(ACCESSOR& self, np::ndarray& np_buffer);

    static bool write(ACCESSOR& self, np::ndarray& np_buffer);

    static bool writeDestructively(ACCESSOR& self, np::ndarray& np_buffer);
  };

  /*****************************************************************************************************************/

  /**
   * (Static) class implementing special functions for Void register accessors
   */
  class VoidRegisterAccessor {
   public:
    static bool write(ctk::VoidRegisterAccessor& self);

    static void read(ctk::VoidRegisterAccessor& self);

    static bool readNonBlocking(ctk::VoidRegisterAccessor& self);

    static bool readLatest(ctk::VoidRegisterAccessor& self);

    static bool writeDestructively(ctk::VoidRegisterAccessor& self);
  };

  /*****************************************************************************************************************/

  class VersionNumber {
   public:
    static boost::posix_time::ptime getTime(ctk::VersionNumber& self);
    static ctk::VersionNumber getNullVersion();
  };

  /*****************************************************************************************************************/

  /**
   * Map RegisterCatalogue class to avoid dealing with RegisterPath objects in Python
   */
  class RegisterCatalogue {
   public:
    static bool hasRegister(ctk::RegisterCatalogue& self, const std::string& registerPathName);
    static ctk::RegisterInfo getRegister(ctk::RegisterCatalogue& self, const std::string& registerPathName);
  };

  /*****************************************************************************************************************/

  class RegisterInfo {
   public:
    // Translate return type from RegisterPath to string
    static std::string getRegisterName(ctk::RegisterInfo& self);

    // convert return type form ChimeraTK::AccessModeFlags to Python list
    static boost::python::list getSupportedAccessModes(ctk::RegisterInfo& self);
  };

  /*****************************************************************************************************************/
  /* Free helper functions */
  /*****************************************************************************************************************/

  ctk::DataType convert_dytpe_to_usertype(np::dtype dtype);

  np::dtype convert_usertype_to_dtype(ctk::DataType usertype);

  template<typename T>
  np::ndarray copyUserBufferToNpArray(ctk::NDRegisterAccessorAbstractor<T>& self, const np::dtype& dtype, size_t ndim);

  template<typename T>
  np::ndarray copyUserBufferToNpArray(ctk::NDRegisterAccessorAbstractor<T>& self, np::ndarray& np_buffer) {
    return copyUserBufferToNpArray<T>(self, np_buffer.get_dtype(), np_buffer.get_nd());
  }

  std::string convertStringFromPython(size_t linearIndex, np::ndarray& np_buffer);

  template<typename T>
  void copyNpArrayToUserBuffer(ctk::NDRegisterAccessorAbstractor<T>& self, np::ndarray& np_buffer);

  /*****************************************************************************************************************/
  /*****************************************************************************************************************/
  /* Implementations following */
  /*****************************************************************************************************************/
  /*****************************************************************************************************************/

  template<typename ACCESSOR>
  np::ndarray GeneralRegisterAccessor<ACCESSOR>::read(ACCESSOR& self, np::ndarray& np_buffer) {
    self.read();
    return copyUserBufferToNpArray(self, np_buffer);
  }

  /*****************************************************************************************************************/

  template<typename ACCESSOR>
  auto GeneralRegisterAccessor<ACCESSOR>::readNonBlocking(ACCESSOR& self, np::ndarray& np_buffer) {
    bool status = self.readNonBlocking();
    return p::make_tuple(status, copyUserBufferToNpArray(self, np_buffer));
  }

  /*****************************************************************************************************************/

  template<typename ACCESSOR>
  auto GeneralRegisterAccessor<ACCESSOR>::readLatest(ACCESSOR& self, np::ndarray& np_buffer) {
    bool status = self.readLatest();
    return p::make_tuple(status, copyUserBufferToNpArray(self, np_buffer));
  }

  /*****************************************************************************************************************/

  template<typename ACCESSOR>
  bool GeneralRegisterAccessor<ACCESSOR>::write(ACCESSOR& self, np::ndarray& np_buffer) {
    copyNpArrayToUserBuffer(self, np_buffer);
    return self.write();
  }

  /*****************************************************************************************************************/

  template<typename ACCESSOR>
  bool GeneralRegisterAccessor<ACCESSOR>::writeDestructively(ACCESSOR& self, np::ndarray& np_buffer) {
    copyNpArrayToUserBuffer(self, np_buffer);
    return self.writeDestructively();
  }

  /*****************************************************************************************************************/

  template<typename T>
  np::ndarray copyUserBufferToNpArray(ctk::NDRegisterAccessorAbstractor<T>& self, const np::dtype& dtype, size_t ndim) {
    auto acc = boost::static_pointer_cast<ctk::NDRegisterAccessor<T>>(self.getHighLevelImplElement());
    auto channels = acc->getNumberOfChannels();
    auto elements = acc->getNumberOfSamples();

    // create new numpy ndarray with proper type
    np::dtype newdtype = dtype; // not a string: keep dtype unchanged
    if constexpr(std::is_same<T, std::string>::value) {
      // string: find longest string in user buffer and set type to unicode string of that length
      size_t neededlength = 0;
      for(size_t i = 0; i < channels; ++i) {
        for(size_t k = 0; k < elements; ++k) {
          neededlength = std::max(acc->accessChannel(i)[k].length(), neededlength);
        }
      }
      newdtype = np::dtype(p::make_tuple("U", neededlength));
    }

    // note: keeping the original shape is important, as we need to distinguish a 2D accessor with 1 channel from
    // a 1D accessor etc.
    assert(ndim <= 2);
    auto new_buffer = ndim == 0 ? np::empty(p::make_tuple(1), newdtype) :
                                  (ndim == 1 ? np::empty(p::make_tuple(elements), newdtype) :
                                               np::empty(p::make_tuple(channels, elements), newdtype));

    // copy data into the mumpy ndarray
    if(ndim <= 1) {
      for(size_t k = 0; k < elements; ++k) {
        if constexpr(std::is_same<T, ctk::Boolean>::value) {
          new_buffer[k] = bool(acc->accessChannel(0)[k]);
        }
        else {
          new_buffer[k] = T(acc->accessChannel(0)[k]);
        }
      }
    }
    else {
      for(size_t i = 0; i < channels; ++i) {
        for(size_t k = 0; k < elements; ++k) {
          if constexpr(std::is_same<T, ctk::Boolean>::value) {
            new_buffer[i][k] = bool(acc->accessChannel(i)[k]);
          }
          else {
            new_buffer[i][k] = T(acc->accessChannel(i)[k]);
          }
        }
      }
    }
    return new_buffer;
  }

  /*****************************************************************************************************************/

  template<typename T>
  void copyNpArrayToUserBuffer(ctk::NDRegisterAccessorAbstractor<T>& self, np::ndarray& np_buffer) {
    auto acc = boost::static_pointer_cast<ctk::NDRegisterAccessor<T>>(self.getHighLevelImplElement());
    auto channels = acc->getNumberOfChannels();
    auto elements = acc->getNumberOfSamples();

    size_t itemsize = np_buffer.get_dtype().get_itemsize();

    if constexpr(!std::is_same<T, std::string>::value) {
      // This check does not work for std::string and is not needed there
      assert(sizeof(*acc->accessChannel(0).data()) == itemsize);
    }
    assert(np_buffer.get_nd() == 2 ? (np_buffer.shape(0) == channels && np_buffer.shape(1) == elements) :
                                     (np_buffer.get_nd() == 1 ? (np_buffer.shape(0) == elements) : elements == 1));

    for(size_t i = 0; i < channels; ++i) {
      if constexpr(std::is_same<T, std::string>::value) {
        for(size_t k = 0; k < elements; ++k) {
          acc->accessChannel(i)[k] = convertStringFromPython(elements * i + k, np_buffer);
        }
      }
      else {
        memcpy(acc->accessChannel(i).data(), np_buffer.get_data() + itemsize * elements * i, itemsize * elements);
      }
    }
  }

  /*****************************************************************************************************************/

} // namespace mtca4upy
