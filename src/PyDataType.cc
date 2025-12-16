// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyDataType.h"

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/

  /********************************************************************************************************************/

  void ChimeraTK::PyDataType::bind(py::module& mod) {
    py::class_<ChimeraTK::DataType> mDataType(mod, "DataType",
        R"(The actual enum representing the data type.
          It is a plain enum so the data type class can be used like a class enum,
          i.e. types are identified for instance as DataType::int32.)");
    mDataType.def(py::init<ChimeraTK::DataType::TheType>())
        .def("__str__", &ChimeraTK::DataType::getAsString)
        .def("__repr__", [](const ChimeraTK::DataType& type) { return "DataType." + type.getAsString(); })
        .def("isNumeric", &ChimeraTK::DataType::isNumeric,
            R"(Returns whether the data type is numeric. Type 'none' returns false.

              :return: True if the data type is numeric, false otherwise.
              :rtype: bool)")
        .def("getAsString", &ChimeraTK::DataType::getAsString,
            R"(Get the data type as string.

              :return: Data type as string.
              :rtype: str)")
        .def("isIntegral", &ChimeraTK::DataType::isIntegral,
            R"(Return whether the raw data type is an integer. False is also returned for non-numerical types and 'none'.

              :return: True if the data type is an integer, false otherwise.
              :rtype: bool)")
        .def("isSigned", &ChimeraTK::DataType::isSigned,
            R"(Return whether the raw data type is signed. True for signed integers and floating point types (currently only signed implementations). False otherwise (also for non-numerical types and 'none').

              :return: True if the data type is signed, false otherwise.
              :rtype: bool)");

    py::enum_<ChimeraTK::DataType::TheType>(mDataType, "TheType")
        .value("none", ChimeraTK::DataType::none,
            "The data type/concept does not exist, e.g. there is no raw transfer (do not confuse with Void)")
        .value("int8", ChimeraTK::DataType::int8)
        .value("uint8", ChimeraTK::DataType::uint8)
        .value("int16", ChimeraTK::DataType::int16)
        .value("uint16", ChimeraTK::DataType::uint16)
        .value("int32", ChimeraTK::DataType::int32)
        .value("uint32", ChimeraTK::DataType::uint32)
        .value("int64", ChimeraTK::DataType::int64)
        .value("uint64", ChimeraTK::DataType::uint64)
        .value("float32", ChimeraTK::DataType::float32)
        .value("float64", ChimeraTK::DataType::float64)
        .value("string", ChimeraTK::DataType::string)
        .value("Boolean", ChimeraTK::DataType::Boolean)
        .value("Void", ChimeraTK::DataType::Void)
        .export_values();
    py::implicitly_convertible<ChimeraTK::DataType::TheType, ChimeraTK::DataType>();
    py::implicitly_convertible<ChimeraTK::DataType, ChimeraTK::DataType::TheType>();
  }

  /********************************************************************************************************************/

  /********************************************************************************************************************/

} // namespace ChimeraTK
