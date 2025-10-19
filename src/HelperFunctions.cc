// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "HelperFunctions.h"

#include "PyVersionNumber.h"

#include <pybind11/embed.h>

#include <string>

namespace py = pybind11;
using namespace py::literals;

namespace ChimeraTK {

  /*****************************************************************************************************************/

  ChimeraTK::DataType convertDTypeToUsertype(const py::dtype& dtype) {
    if(dtype.is(py::dtype::of<int8_t>())) {
      return ChimeraTK::DataType::int8;
    }
    if(dtype.is(py::dtype::of<int16_t>())) {
      return ChimeraTK::DataType::int16;
    }
    if(dtype.is(py::dtype::of<int32_t>())) {
      return ChimeraTK::DataType::int32;
    }
    if(dtype.is(py::dtype::of<int64_t>())) {
      return ChimeraTK::DataType::int64;
    }
    if(dtype.is(py::dtype::of<uint8_t>())) {
      return ChimeraTK::DataType::uint8;
    }
    if(dtype.is(py::dtype::of<uint16_t>())) {
      return ChimeraTK::DataType::uint16;
    }
    if(dtype.is(py::dtype::of<uint32_t>())) {
      return ChimeraTK::DataType::uint32;
    }
    if(dtype.is(py::dtype::of<uint64_t>())) {
      return ChimeraTK::DataType::uint64;
    }
    if(dtype.is(py::dtype::of<float>())) {
      return ChimeraTK::DataType::float32;
    }
    if(dtype.is(py::dtype::of<double>())) {
      return ChimeraTK::DataType::float64;
    }
    if(dtype.is(py::dtype::of<bool>())) {
      return ChimeraTK::DataType::Boolean;
    }
    if(dtype.kind() == 'U') { // Unicode string
      return ChimeraTK::DataType::string;
    }
    if(dtype.kind() == 'S') { // ASCII bytes string
      return ChimeraTK::DataType::string;
    }
    throw std::invalid_argument("Unsupported numpy dtype");
  }

  /*****************************************************************************************************************/

  py::dtype convertUsertypeToDtype(const ChimeraTK::DataType& usertype) {
    std::unique_ptr<py::dtype> rv;
    ChimeraTK::callForTypeNoVoid(usertype, [&](auto arg) {
      using UserType = decltype(arg);
      if constexpr(std::is_same<UserType, ChimeraTK::Boolean>::value) {
        rv = std::make_unique<py::dtype>(py::dtype::of<bool>());
      }
      else if constexpr(std::is_same<UserType, std::string>::value) {
        rv = std::make_unique<py::dtype>(py::dtype::of<char*>());
      }
      else {
        rv = std::make_unique<py::dtype>(py::dtype::of<UserType>());
      }
    });
    return *rv;
  }

  /*****************************************************************************************************************/

  PyVersionNumber getNewVersionNumberIfNull(const PyVersionNumber& versionNumber) {
    // If the version number is null, we return a new version number.
    // Otherwise, we return the given version number.
    if(versionNumber == PyVersionNumber::getNullVersion()) {
      return PyVersionNumber();
    }
    return versionNumber;
  }

  /*****************************************************************************************************************/

  py::array convertPyListToNumpyArray(const py::list& list, const py::dtype& dtype) {
    py::gil_scoped_acquire gil;
    auto locals = py::dict("incommingdtype"_a = dtype, "incommingList"_a = list);
    py::exec(R"(
        import numpy as np
        out = np.array(incommingList, dtype=incommingdtype)
    )",
        py::globals(), locals);
    return locals["out"].cast<py::array>();
  }

  /*****************************************************************************************************************/

  [[nodiscard]] py::list accessModeFlagsToList(const ChimeraTK::AccessModeFlags& flags) {
    py::list python_flags{};
    if(flags.has(ChimeraTK::AccessMode::raw)) {
      python_flags.append(ChimeraTK::AccessMode::raw);
    }
    if(flags.has(ChimeraTK::AccessMode::wait_for_new_data)) {
      python_flags.append(ChimeraTK::AccessMode::wait_for_new_data);
    }
    return python_flags;
  }

  /*****************************************************************************************************************/

} // namespace ChimeraTK
