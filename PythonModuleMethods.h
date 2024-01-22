#ifndef REGISTERACCESSORWRAPPERFUNCTIONS_H_
#define REGISTERACCESSORWRAPPERFUNCTIONS_H_

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

namespace boost::python::numpy::detail {

  /**
   * Provide numpy dtype for ChimeraTK::Boolean (conversion is identical as for plain bool)
   */
  template<>
  struct builtin_dtype<ChimeraTK::Boolean, false> {
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
 * Provide converter from ChimeraTK::Boolean into Python bool type
 */
struct CtkBoolean_to_python {
  static PyObject* convert(ChimeraTK::Boolean const& value) {
    return boost::python::incref(boost::python::object(bool(value)).ptr());
  }
};

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
    std::string getName(T& self) {
      return self.getName();
    }

    template<typename T>
    std::string getUnit(T& self) {
      return self.getUnit();
    }

    template<typename T>
    std::string getDescription(T& self) {
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

    template<typename T>
    std::string getAccessModeFlagsString(T& self) {
      return self.getAccessModeFlags().serialize();
    }

    template<typename T>
    np::ndarray copyUserBufferToNpArray(
        ChimeraTK::NDRegisterAccessorAbstractor<T>& self, const np::dtype& dtype, size_t ndim) {
      auto acc = boost::static_pointer_cast<ChimeraTK::NDRegisterAccessor<T>>(self.getHighLevelImplElement());
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
          if constexpr(std::is_same<T, ChimeraTK::Boolean>::value) {
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
            if constexpr(std::is_same<T, ChimeraTK::Boolean>::value) {
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

    template<typename T>
    np::ndarray copyUserBufferToNpArray(ChimeraTK::NDRegisterAccessorAbstractor<T>& self, np::ndarray& np_buffer) {
      return copyUserBufferToNpArray<T>(self, np_buffer.get_dtype(), np_buffer.get_nd());
    }

    inline std::string convertStringFromPython(size_t linearIndex, np::ndarray& np_buffer) {
      // Note: it is unclear why the conversion in this direction has to be so complicated, while in the other
      // direction an assignment to an std::string is sufficient.

      // create C++ 4-byte string of matching length
      size_t itemsize = np_buffer.get_dtype().get_itemsize();
      assert(itemsize % sizeof(char32_t) == 0);
      std::u32string widestring;
      widestring.resize(itemsize / 4);

      // copy string to C++ buffer
      memcpy(widestring.data(), np_buffer.get_data() + itemsize * linearIndex, itemsize);

      // convert to UTF-8 string and store to accessor
      std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
      return conv.to_bytes(widestring);
    }

    template<typename T>
    void copyNpArrayToUserBuffer(ChimeraTK::NDRegisterAccessorAbstractor<T>& self, np::ndarray& np_buffer) {
      auto acc = boost::static_pointer_cast<ChimeraTK::NDRegisterAccessor<T>>(self.getHighLevelImplElement());
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

    template<typename ACCESSOR>
    np::ndarray read(ACCESSOR& self, np::ndarray& np_buffer) {
      self.read();
      return copyUserBufferToNpArray(self, np_buffer);
    }

    template<typename ACCESSOR>
    auto readNonBlocking(ACCESSOR& self, np::ndarray& np_buffer) {
      bool status = self.readNonBlocking();
      return p::make_tuple(status, copyUserBufferToNpArray(self, np_buffer));
    }

    template<typename ACCESSOR>
    auto readLatest(ACCESSOR& self, np::ndarray& np_buffer) {
      bool status = self.readLatest();
      return p::make_tuple(status, copyUserBufferToNpArray(self, np_buffer));
    }

    template<typename ACCESSOR>
    bool write(ACCESSOR& self, np::ndarray& np_buffer) {
      copyNpArrayToUserBuffer(self, np_buffer);
      return self.write();
    }

    template<typename ACCESSOR>
    bool writeDestructively(ACCESSOR& self, np::ndarray& np_buffer) {
      copyNpArrayToUserBuffer(self, np_buffer);
      return self.writeDestructively();
    }

  } // namespace GeneralRegisterAccessor

  namespace VoidRegisterAccessor {

    bool write(ChimeraTK::VoidRegisterAccessor& self);

    void read(ChimeraTK::VoidRegisterAccessor& self);

    bool readNonBlocking(ChimeraTK::VoidRegisterAccessor& self);

    bool readLatest(ChimeraTK::VoidRegisterAccessor& self);

    bool writeDestructively(ChimeraTK::VoidRegisterAccessor& self);

  } // namespace VoidRegisterAccessor

  namespace ScalarRegisterAccessor {

    template<typename T>
    T readAndGet(ChimeraTK::ScalarRegisterAccessor<T>& self) {
      return self.readAndGet();
    }

    template<typename T>
    void setAndWrite(ChimeraTK::ScalarRegisterAccessor<T>& self, T newValue, ChimeraTK::VersionNumber versionNumber) {
      if(versionNumber == ChimeraTK::VersionNumber(nullptr)) {
        versionNumber = ChimeraTK::VersionNumber();
      }
      self.setAndWrite(newValue, versionNumber);
    }

    template<typename T>
    void writeIfDifferent(
        ChimeraTK::ScalarRegisterAccessor<T>& self, T newValue, ChimeraTK::VersionNumber versionNumber) {
      self.writeIfDifferent(newValue, versionNumber);
    }

  } // namespace ScalarRegisterAccessor

  namespace OneDRegisterAccessor {

    template<typename T>
    int getNElements(ChimeraTK::OneDRegisterAccessor<T>& self) {
      return self.getNElements();
    }

  } // namespace OneDRegisterAccessor

  namespace TwoDRegisterAccessor {

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
    np::ndarray read(const ChimeraTK::Device& self, const std::string& registerPath, size_t numberOfElements,
        size_t elementsOffset, boost::python::list flaglist);
    ChimeraTK::DataType convert_dytpe_to_usertype(np::dtype dtype);
    np::dtype convert_usertype_to_dtype(ChimeraTK::DataType usertype);

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
