// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <ChimeraTK/NDRegisterAccessorAbstractor.h>
#include <ChimeraTK/SupportedUserTypes.h>

#include <boost/python/numpy.hpp>

#include <codecvt>
#include <locale>

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

  /*****************************************************************************************************************/

  ChimeraTK::DataType convert_dytpe_to_usertype(boost::python::numpy::dtype dtype);

  boost::python::numpy::dtype convert_usertype_to_dtype(ChimeraTK::DataType usertype);

  template<typename T>
  boost::python::numpy::ndarray copyUserBufferToNpArray(
      ChimeraTK::NDRegisterAccessorAbstractor<T>& self, const boost::python::numpy::dtype& dtype, size_t ndim);

  template<typename T>
  boost::python::numpy::ndarray copyUserBufferToNpArray(
      ChimeraTK::NDRegisterAccessorAbstractor<T>& self, boost::python::numpy::ndarray& np_buffer) {
    return copyUserBufferToNpArray<T>(self, np_buffer.get_dtype(), np_buffer.get_nd());
  }

  std::string convertStringFromPython(size_t linearIndex, boost::python::numpy::ndarray& np_buffer);

  template<typename T>
  void copyNpArrayToUserBuffer(
      ChimeraTK::NDRegisterAccessorAbstractor<T>& self, boost::python::numpy::ndarray& np_buffer);

  /*****************************************************************************************************************/
  /*****************************************************************************************************************/
  /* Implementations following */
  /*****************************************************************************************************************/
  /*****************************************************************************************************************/

  template<typename T>
  boost::python::numpy::ndarray copyUserBufferToNpArray(
      ChimeraTK::NDRegisterAccessorAbstractor<T>& self, const boost::python::numpy::dtype& dtype, size_t ndim) {
    auto acc = boost::static_pointer_cast<ChimeraTK::NDRegisterAccessor<T>>(self.getHighLevelImplElement());
    auto channels = acc->getNumberOfChannels();
    auto elements = acc->getNumberOfSamples();

    // create new numpy ndarray with proper type
    boost::python::numpy::dtype newdtype = dtype; // not a string: keep dtype unchanged
    if constexpr(std::is_same<T, std::string>::value) {
      // string: find longest string in user buffer and set type to unicode string of that length
      size_t neededlength = 0;
      for(size_t i = 0; i < channels; ++i) {
        for(size_t k = 0; k < elements; ++k) {
          neededlength = std::max(acc->accessChannel(i)[k].length(), neededlength);
        }
      }
      newdtype = boost::python::numpy::dtype(boost::python::make_tuple("U", neededlength));
    }

    // note: keeping the original shape is important, as we need to distinguish a 2D accessor with 1 channel from
    // a 1D accessor etc.
    assert(ndim <= 2);
    auto new_buffer = ndim == 0 ?
        boost::python::numpy::empty(boost::python::make_tuple(1), newdtype) :
        (ndim == 1 ? boost::python::numpy::empty(boost::python::make_tuple(elements), newdtype) :
                     boost::python::numpy::empty(boost::python::make_tuple(channels, elements), newdtype));

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

  /*****************************************************************************************************************/

  template<typename T>
  void copyNpArrayToUserBuffer(
      ChimeraTK::NDRegisterAccessorAbstractor<T>& self, boost::python::numpy::ndarray& np_buffer) {
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

  /*****************************************************************************************************************/

} // namespace mtca4upy
