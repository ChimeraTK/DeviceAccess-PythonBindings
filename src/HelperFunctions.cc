// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "HelperFunctions.h"

#include <ChimeraTK/SupportedUserTypes.h>

#include <boost/python/numpy.hpp>

namespace DeviceAccessPython {

  /*****************************************************************************************************************/

  ChimeraTK::DataType convert_dytpe_to_usertype(boost::python::numpy::dtype dtype) {
    if(dtype == boost::python::numpy::dtype::get_builtin<int8_t>()) {
      return ChimeraTK::DataType::int8;
    }
    if(dtype == boost::python::numpy::dtype::get_builtin<int16_t>()) {
      return ChimeraTK::DataType::int16;
    }
    if(dtype == boost::python::numpy::dtype::get_builtin<int32_t>()) {
      return ChimeraTK::DataType::int32;
    }
    if(dtype == boost::python::numpy::dtype::get_builtin<int64_t>()) {
      return ChimeraTK::DataType::int64;
    }
    if(dtype == boost::python::numpy::dtype::get_builtin<uint8_t>()) {
      return ChimeraTK::DataType::uint8;
    }
    if(dtype == boost::python::numpy::dtype::get_builtin<uint16_t>()) {
      return ChimeraTK::DataType::uint16;
    }
    if(dtype == boost::python::numpy::dtype::get_builtin<uint32_t>()) {
      return ChimeraTK::DataType::uint32;
    }
    if(dtype == boost::python::numpy::dtype::get_builtin<uint64_t>()) {
      return ChimeraTK::DataType::uint64;
    }
    if(dtype == boost::python::numpy::dtype::get_builtin<float>()) {
      return ChimeraTK::DataType::float32;
    }
    if(dtype == boost::python::numpy::dtype::get_builtin<double>()) {
      return ChimeraTK::DataType::float64;
    }
    if(dtype == boost::python::numpy::dtype::get_builtin<bool>()) {
      return ChimeraTK::DataType::Boolean;
    }
    throw std::invalid_argument("Unsupported numpy dtype");
  }

  /*****************************************************************************************************************/

  boost::python::numpy::dtype convert_usertype_to_dtype(ChimeraTK::DataType usertype) {
    std::unique_ptr<boost::python::numpy::dtype> rv;
    ChimeraTK::callForTypeNoVoid(usertype, [&](auto arg) {
      using UserType = decltype(arg);
      if constexpr(std::is_same<UserType, ChimeraTK::Boolean>::value) {
        rv = std::make_unique<boost::python::numpy::dtype>(boost::python::numpy::dtype::get_builtin<bool>());
      }
      if constexpr(std::is_same<UserType, std::string>::value) {
        rv = std::make_unique<boost::python::numpy::dtype>(boost::python::make_tuple('U', 1));
      }
      else {
        rv = std::make_unique<boost::python::numpy::dtype>(boost::python::numpy::dtype::get_builtin<UserType>());
      }
    });
    return *rv;
  }

  /*****************************************************************************************************************/

  std::string convertStringFromPython(size_t linearIndex, boost::python::numpy::ndarray& np_buffer) {
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

  /*****************************************************************************************************************/

} // namespace DeviceAccessPython
