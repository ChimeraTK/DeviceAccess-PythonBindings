// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <pybind11/pybind11.h>
// pybind11.h must come first

#include "PyDataType.h"
#include "PyDevice.h"
#include "PyOneDRegisterAccessor.h"
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

PYBIND11_MODULE(deviceaccess, m) {
  ChimeraTK::PyDevice::bind(m);
  ChimeraTK::PyVersionNumber::bind(m);
  ChimeraTK::PyTransferElementBase::bind(m);
  ChimeraTK::PyScalarRegisterAccessor::bind(m);
  ChimeraTK::PyTwoDRegisterAccessor::bind(m);
  ChimeraTK::PyOneDRegisterAccessor::bind(m);
  ChimeraTK::PyVoidRegisterAccessor::bind(m);
  ChimeraTK::PyDataType::bind(m);

  m.def("setDMapFilePath", ChimeraTK::setDMapFilePath);
  m.def("getDMapFilePath", ChimeraTK::getDMapFilePath);

  py::class_<ChimeraTK::RegisterCatalogue>(m, "RegisterCatalogue")
      .def(py::init<ChimeraTK::RegisterCatalogue>())
      .def(
          "__iter__", [](const ChimeraTK::RegisterCatalogue& s) { return py::make_iterator(s.begin(), s.end()); },
          py::keep_alive<0, 1>())
      .def("_items", DeviceAccessPython::RegisterCatalogue::items)
      .def("hiddenRegisters", DeviceAccessPython::RegisterCatalogue::hiddenRegisters)
      .def("hasRegister", &ChimeraTK::RegisterCatalogue::hasRegister)
      .def("getRegister", &ChimeraTK::RegisterCatalogue::getRegister);

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

  py::class_<ChimeraTK::BackendRegisterInfoBase>(m, "BackendRegisterInfoBase")
      .def("getDataDescriptor", &ChimeraTK::BackendRegisterInfoBase::getDataDescriptor)
      .def("isReadable", &ChimeraTK::BackendRegisterInfoBase::isReadable)
      .def("isWriteable", &ChimeraTK::BackendRegisterInfoBase::isWriteable)
      .def("getRegisterName", &ChimeraTK::BackendRegisterInfoBase::getRegisterName)
      .def("getSupportedAccessModes", &ChimeraTK::BackendRegisterInfoBase::getSupportedAccessModes)
      .def("getNumberOfElements", &ChimeraTK::BackendRegisterInfoBase::getNumberOfElements)
      .def("getNumberOfDimensions", &ChimeraTK::BackendRegisterInfoBase::getNumberOfDimensions)
      .def("getNumberOfChannels", &ChimeraTK::BackendRegisterInfoBase::getNumberOfChannels);

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

  py::class_<ChimeraTK::RegisterPath>(m, "RegisterPath")
      .def(py::init<ChimeraTK::RegisterPath>())
      .def(py::init<const std::string&>(), py::arg("s"))
      .def("__str__", &ChimeraTK::RegisterPath::operator std::string)
      .def("setAltSeparator", &ChimeraTK::RegisterPath::setAltSeparator)
      .def("getWithAltSeparator", &ChimeraTK::RegisterPath::getWithAltSeparator)
      .def("__itruediv__", &ChimeraTK::RegisterPath::operator/=)
      .def("__iadd__", &ChimeraTK::RegisterPath::operator+=)
      .def("__lt__", &ChimeraTK::RegisterPath::operator<)
      .def("length", &ChimeraTK::RegisterPath::length)
      .def("startsWith", &ChimeraTK::RegisterPath::startsWith)
      .def("endsWith", &ChimeraTK::RegisterPath::endsWith)
      .def("getComponents", &ChimeraTK::RegisterPath::getComponents)
      .def("__ne__",
          [](const ChimeraTK::RegisterPath& self, const ChimeraTK::RegisterPath& other) { return self != other; })
      .def("__ne__", [](const ChimeraTK::RegisterPath& self, const std::string& other) { return self != other; })
      .def("__eq__",
          [](const ChimeraTK::RegisterPath& self, const ChimeraTK::RegisterPath& other) { return self == other; })
      .def("__eq__", [](const ChimeraTK::RegisterPath& self, const std::string& other) { return self == other; });

  py::implicitly_convertible<std::string, ChimeraTK::RegisterPath>();

  // Map the boost::thread_interrupted exception. We cannot use py::register_exception because there is no what().
  static py::exception<boost::thread_interrupted> exc(m, "ThreadInterrupted");
  // NOLINTNEXTLINE(performance-unnecessary-value-param) - signature with reference not accepted here...
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if(p) {
        std::rethrow_exception(p);
      }
    }
    catch(const boost::thread_interrupted&) {
      exc("Thread Interrupted");
    }
  });
}
