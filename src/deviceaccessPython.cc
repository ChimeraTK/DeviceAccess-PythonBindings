// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyDevice.h"
#include "PyOneDRegisterAccessor.h"
#include "PythonModuleMethods.h"
#include "PyTwoDRegisterAccessor.h"
#include "RegisterCatalogue.h"
#include "VersionNumber.h"

#include <ChimeraTK/SupportedUserTypes.h>

#include <pybind11/pybind11.h>
namespace py = pybind11;

//****************************************************************************//
// Holder definitions
PYBIND11_DECLARE_HOLDER_TYPE(T, boost::shared_ptr<T>)

//****************************************************************************//

PYBIND11_MODULE(_da_python_bindings, m) {
  ChimeraTK::PyDevice::bind(m);
  ChimeraTK::PyTransferElementBase::bind(m);
  ChimeraTK::PyScalarRegisterAccessor::bind(m);
  ChimeraTK::PyTwoDRegisterAccessor::bind(m);
  ChimeraTK::PyOneDRegisterAccessor::bind(m);
  ChimeraTK::PyVoidRegisterAccessor::bind(m);

  /**
   *  DataType (with internal enum)
   */
  py::class_<ChimeraTK::DataType> mDataType(m, "DataType");
  mDataType.def(py::init<ChimeraTK::DataType::TheType>())
      .def("__str__", &ChimeraTK::DataType::getAsString)
      // TODO: add __eq__ and __ne__ to DataType
      //.def(py::self == py::self)
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

  m.def("createDevice", DeviceAccessPython::createDevice);
  m.def("getDevice_no_alias", DeviceAccessPython::getDevice_no_alias);
  m.def("getDevice", DeviceAccessPython::getDevice);
  m.def("setDMapFilePath", DeviceAccessPython::setDmapFile);
  m.def("getDMapFilePath", DeviceAccessPython::getDmapFile);

  py::class_<ChimeraTK::RegisterCatalogue>(m, "RegisterCatalogue")
      .def(py::init<ChimeraTK::RegisterCatalogue>())
      .def("_items", DeviceAccessPython::RegisterCatalogue::items)
      .def("hasRegister", DeviceAccessPython::RegisterCatalogue::hasRegister)
      .def("getRegister", DeviceAccessPython::RegisterCatalogue::getRegister);

  py::class_<ChimeraTK::RegisterInfo>(m, "RegisterInfo")
      .def(py::init<ChimeraTK::RegisterInfo>())
      .def("getDataDescriptor", DeviceAccessPython::RegisterInfo::getDataDescriptor)
      .def("isReadable", &ChimeraTK::RegisterInfo::isReadable)
      .def("isValid", &ChimeraTK::RegisterInfo::isValid)
      .def("isWriteable", &ChimeraTK::RegisterInfo::isWriteable)
      .def("getRegisterName", DeviceAccessPython::RegisterInfo::getRegisterName)
      .def("getSupportedAccessModes", DeviceAccessPython::RegisterInfo::getSupportedAccessModes)
      .def("getNumberOfElements", &ChimeraTK::RegisterInfo::getNumberOfElements)
      .def("getNumberOfDimensions", &ChimeraTK::RegisterInfo::getNumberOfDimensions)
      .def("getNumberOfChannels", &ChimeraTK::RegisterInfo::getNumberOfChannels);

  py::class_<ChimeraTK::DataDescriptor>(m, "DataDescriptor")
      .def(py::init<ChimeraTK::DataDescriptor>())
      .def("rawDataType", &ChimeraTK::DataDescriptor::rawDataType)
      .def("transportLayerDataType", &ChimeraTK::DataDescriptor::transportLayerDataType)
      .def("minimumDataType", &ChimeraTK::DataDescriptor::minimumDataType)
      .def("isSigned", &ChimeraTK::DataDescriptor::isSigned)
      .def("isIntegral", &ChimeraTK::DataDescriptor::isIntegral)
      .def("nDigits", &ChimeraTK::DataDescriptor::nDigits)
      .def("nFractionalDigits", &ChimeraTK::DataDescriptor::nFractionalDigits)
      .def("fundamentalType", DeviceAccessPython::DataDescriptor::fundamentalType);

  py::enum_<ChimeraTK::AccessMode>(m, "AccessMode")
      .value("raw", ChimeraTK::AccessMode::raw)
      .value("wait_for_new_data", ChimeraTK::AccessMode::wait_for_new_data)
      .export_values();

  py::enum_<ChimeraTK::DataDescriptor::FundamentalType>(m, "FundamentalType")
      .value("numeric", ChimeraTK::DataDescriptor::FundamentalType::numeric)
      .value("string", ChimeraTK::DataDescriptor::FundamentalType::string)
      .value("boolean", ChimeraTK::DataDescriptor::FundamentalType::boolean)
      .value("nodata", ChimeraTK::DataDescriptor::FundamentalType::nodata)
      .value("undefined", ChimeraTK::DataDescriptor::FundamentalType::undefined)
      .export_values();

  py::enum_<ChimeraTK::DataValidity>(m, "DataValidity")
      .value("ok", ChimeraTK::DataValidity::ok)
      .value("faulty", ChimeraTK::DataValidity::faulty)
      .export_values();

  py::class_<ChimeraTK::TransferElementID>(m, "TransferElementID")
      .def("isValid", &ChimeraTK::TransferElementID::isValid)
      .def("__ne__", &ChimeraTK::TransferElementID::operator!=)
      .def("__eq__", &ChimeraTK::TransferElementID::operator==);

  py::class_<ChimeraTK::VersionNumber>(m, "VersionNumber",
      "Class for generating and holding version numbers without exposing a numeric representation.\n"
      "\n"
      "Version numbers are used to resolve competing updates that are applied to the same process variable. For "
      "example, it they can help in breaking an infinite update loop that might occur when two process variables are "
      "related and update each other.\n"
      "\n"
      "They are also used to determine the order of updates made to different process variables.\n"
      "\n")
      .def(py::init<>())
      .def("getTime", &DeviceAccessPython::VersionNumber::getTime)
      .def("__str__", &ChimeraTK::VersionNumber::operator std::string)
      .def("getNullVersion", DeviceAccessPython::VersionNumber::getNullVersion)
      .def("__lt__", &ChimeraTK::VersionNumber::operator<)
      .def("__le__", &ChimeraTK::VersionNumber::operator<=)
      .def("__gt__", &ChimeraTK::VersionNumber::operator>)
      .def("__ge__", &ChimeraTK::VersionNumber::operator>=)
      .def("__ne__", &ChimeraTK::VersionNumber::operator!=)
      .def("__eq__", &ChimeraTK::VersionNumber::operator==);
}
