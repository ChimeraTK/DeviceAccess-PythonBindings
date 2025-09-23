// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyDataType.h"

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/

  /********************************************************************************************************************/

  void ChimeraTK::PyDataType::bind(py::module& mod) {
    py::class_<ChimeraTK::DataType> mDataType(mod, "DataType");
    mDataType.def(py::init<ChimeraTK::DataType::TheType>())
        .def("__str__", &ChimeraTK::DataType::getAsString)
        .def("__repr__", [](const ChimeraTK::DataType& type) { return "DataType." + type.getAsString(); })
        .def("isNumeric", &ChimeraTK::DataType::isNumeric)
        .def("getAsString", &ChimeraTK::DataType::getAsString)
        .def("isIntegral", &ChimeraTK::DataType::isIntegral)
        .def("isSigned", &ChimeraTK::DataType::isSigned);

    py::enum_<ChimeraTK::DataType::TheType>(mDataType, "TheType")
        .value("none", ChimeraTK::DataType::none)
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