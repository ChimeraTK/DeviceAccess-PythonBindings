// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <ChimeraTK/NDRegisterAccessorAbstractor.h>
#include <ChimeraTK/SupportedUserTypes.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <boost/python/numpy.hpp>

#include <codecvt>
#include <locale>
namespace py = pybind11;

namespace PYBIND11_NAMESPACE { namespace detail {
  template<>
  struct type_caster<ChimeraTK::Boolean> {
   public:
    /**
     * This macro establishes the name 'Boolean' in
     * function signatures and declares a local variable
     * 'value' of type ChimeraTK::Boolean
     */
    PYBIND11_TYPE_CASTER(ChimeraTK::Boolean, const_name("Boolean"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into a ChimeraTK::Boolean
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool) {
      /* Extract PyObject from handle */
      PyObject* source = src.ptr();
      /* Try converting into a Python integer value */
      PyObject* tmp = PyNumber_Long(source);
      if(!tmp) return false;
      /* Now try to convert into a C++ int */
      value = bool(PyLong_AsLong(tmp));
      Py_DECREF(tmp);
      return true;
    }

    /**
     * Conversion part 2 (C++ -> Python): convert an inty instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(ChimeraTK::Boolean src, return_value_policy /* policy */, handle /* parent */) {
      return PyLong_FromLong(src);
    }
  };
}} // namespace PYBIND11_NAMESPACE::detail

namespace DeviceAccessPython {

  /*****************************************************************************************************************/

  ChimeraTK::DataType convert_dytpe_to_usertype(py::dtype dtype);

  py::dtype convert_usertype_to_dtype(ChimeraTK::DataType usertype);

  template<typename T>
  pybind11::array copyUserBufferToNpArray(
      ChimeraTK::NDRegisterAccessorAbstractor<T>& self, const pybind11::dtype& dtype, size_t ndim);

  template<typename T>
  pybind11::array copyUserBufferToNpArray(
      ChimeraTK::NDRegisterAccessorAbstractor<T>& self, pybind11::array& np_buffer) {
    return copyUserBufferToNpArray<T>(self, np_buffer.dtype(), np_buffer.ndim());
  }

  std::string convertStringFromPython(size_t linearIndex, pybind11::array& np_buffer);

  template<typename T>
  void copyNpArrayToUserBuffer(ChimeraTK::NDRegisterAccessorAbstractor<T>& self, pybind11::array& np_buffer);

  /*****************************************************************************************************************/
  /*****************************************************************************************************************/
  /* Implementations following */
  /*****************************************************************************************************************/
  /*****************************************************************************************************************/

  template<typename T>
  pybind11::array copyUserBufferToNpArray(
      ChimeraTK::NDRegisterAccessorAbstractor<T>& self, const pybind11::dtype& dtype, size_t ndim) {
    auto acc = boost::static_pointer_cast<ChimeraTK::NDRegisterAccessor<T>>(self.getHighLevelImplElement());
    auto channels = acc->getNumberOfChannels();
    auto elements = acc->getNumberOfSamples();

    // create new numpy ndarray with proper type
    pybind11::dtype newdtype = dtype; // not a string: keep dtype unchanged
    if constexpr(std::is_same<T, std::string>::value) {
      // string: find longest string in user buffer and set type to unicode string of that length
      size_t neededlength = 0;
      for(size_t i = 0; i < channels; ++i) {
        for(size_t k = 0; k < elements; ++k) {
          neededlength = std::max(acc->accessChannel(i)[k].length(), neededlength);
        }
      }
      newdtype = pybind11::dtype("U" + std::to_string(neededlength));
    }

    // note: keeping the original shape is important, as we need to distinguish a 2D accessor with 1 channel from
    // a 1D accessor etc.
    assert(ndim <= 2);
    auto new_buffer = ndim == 0 ?
        pybind11::array(newdtype, pybind11::array::ShapeContainer{pybind11::ssize_t(1)}) :
        (ndim == 1 ? pybind11::array(newdtype, pybind11::array::ShapeContainer{pybind11::ssize_t(elements)}) :
                     pybind11::array(newdtype,
                         pybind11::array::ShapeContainer{pybind11::ssize_t(channels), pybind11::ssize_t(elements)}));

    // copy data into the mumpy ndarray
    if(ndim <= 1) {
      if constexpr(std::is_same<T, std::string>::value) {
        // strings need to be copied per element, since numpy expect them to be organised in consecutive memory
        for(size_t k = 0; k < elements; ++k) {
          std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
          memcpy(new_buffer.mutable_data(k), conv.from_bytes(acc->accessChannel(0)[k]).data(),
              acc->accessChannel(0)[k].length() * 4);
        }
      }
      else {
        memcpy(new_buffer.mutable_data(), acc->accessChannel(0).data(), elements * sizeof(T));
      }
    }
    else {
      for(size_t i = 0; i < channels; ++i) {
        memcpy(new_buffer.mutable_data(i), acc->accessChannel(i).data(), elements * sizeof(T));
      }
    }
    return new_buffer;
  }

  /*****************************************************************************************************************/

  template<typename T>
  void copyNpArrayToUserBuffer(ChimeraTK::NDRegisterAccessorAbstractor<T>& self, pybind11::array& np_buffer) {
    auto acc = boost::static_pointer_cast<ChimeraTK::NDRegisterAccessor<T>>(self.getHighLevelImplElement());
    auto channels = acc->getNumberOfChannels();
    auto elements = acc->getNumberOfSamples();

    size_t itemsize = np_buffer.dtype().itemsize();

    if constexpr(!std::is_same<T, std::string>::value) {
      // This check does not work for std::string and is not needed there
      assert(sizeof(*acc->accessChannel(0).data()) == itemsize);
    }
    assert(np_buffer.ndim() == 2 ? (np_buffer.shape(0) == channels && np_buffer.shape(1) == elements) :
                                   (np_buffer.ndim() == 1 ? (np_buffer.shape(0) == elements) : elements == 1));

    for(size_t i = 0; i < channels; ++i) {
      if constexpr(std::is_same<T, std::string>::value) {
        for(size_t k = 0; k < elements; ++k) {
          acc->accessChannel(i)[k] = convertStringFromPython(elements * i + k, np_buffer);
        }
      }
      else {
        memcpy(acc->accessChannel(i).data(), np_buffer.data(i), itemsize * elements);
      }
    }
  }

  /*****************************************************************************************************************/

} // namespace DeviceAccessPython
