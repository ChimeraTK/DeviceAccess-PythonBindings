// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <pybind11/pybind11.h>
// pybind11.h must come first

#include "PyDataType.h"
#include "PyDevice.h"
#include "PyOneDRegisterAccessor.h"
#include "PythonModuleMethods.h"
#include "PyTwoDRegisterAccessor.h"
#include "PyVersionNumber.h"
#include "RegisterCatalogue.h"

#include <ChimeraTK/SupportedUserTypes.h>
#include <ChimeraTK/VariantUserTypes.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

//****************************************************************************//
// Holder definitions
PYBIND11_DECLARE_HOLDER_TYPE(T, boost::shared_ptr<T>)

//****************************************************************************//

PYBIND11_MODULE(_da_python_bindings, m) {
  ChimeraTK::PyDevice::bind(m);
  ChimeraTK::PyVersionNumber::bind(m);
  ChimeraTK::PyTransferElementBase::bind(m);
  ChimeraTK::PyScalarRegisterAccessor::bind(m);
  ChimeraTK::PyTwoDRegisterAccessor::bind(m);
  ChimeraTK::PyOneDRegisterAccessor::bind(m);
  ChimeraTK::PyVoidRegisterAccessor::bind(m);
  ChimeraTK::PyDataType::bind(m);

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
}
