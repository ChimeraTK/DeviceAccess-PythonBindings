// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "HelperFunctions.h"

#include <ChimeraTK/SupportedUserTypes.h>

namespace py = pybind11;

namespace DeviceAccessPython {

  /*****************************************************************************************************************/

  ChimeraTK::DataType convert_dytpe_to_usertype(py::dtype dtype) {
    if(dtype == py::dtype::of<int8_t>()) {
      return ChimeraTK::DataType::int8;
    }
    if(dtype == py::dtype::of<int16_t>()) {
      return ChimeraTK::DataType::int16;
    }
    if(dtype == py::dtype::of<int32_t>()) {
      return ChimeraTK::DataType::int32;
    }
    if(dtype == py::dtype::of<int64_t>()) {
      return ChimeraTK::DataType::int64;
    }
    if(dtype == py::dtype::of<uint8_t>()) {
      return ChimeraTK::DataType::uint8;
    }
    if(dtype == py::dtype::of<uint16_t>()) {
      return ChimeraTK::DataType::uint16;
    }
    if(dtype == py::dtype::of<uint32_t>()) {
      return ChimeraTK::DataType::uint32;
    }
    if(dtype == py::dtype::of<uint64_t>()) {
      return ChimeraTK::DataType::uint64;
    }
    if(dtype == py::dtype::of<float>()) {
      return ChimeraTK::DataType::float32;
    }
    if(dtype == py::dtype::of<double>()) {
      return ChimeraTK::DataType::float64;
    }
    if(dtype == py::dtype::of<bool>()) {
      return ChimeraTK::DataType::Boolean;
    }
    throw std::invalid_argument("Unsupported numpy dtype");
  }

  /*****************************************************************************************************************/

  py::dtype convert_usertype_to_dtype(ChimeraTK::DataType usertype) {
    std::unique_ptr<py::dtype> rv;
    ChimeraTK::callForTypeNoVoid(usertype, [&](auto arg) {
      using UserType = decltype(arg);
      if constexpr(std::is_same<UserType, ChimeraTK::Boolean>::value) {
        rv = std::make_unique<py::dtype>(py::dtype::of<bool>());
      }
      if constexpr(std::is_same<UserType, std::string>::value) {
        rv = std::make_unique<py::dtype>(py::dtype::of<std::string>());
      }
      else {
        rv = std::make_unique<py::dtype>(py::dtype::of<UserType>());
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
