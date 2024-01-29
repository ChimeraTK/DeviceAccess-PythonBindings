// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PythonModuleMethods.h"

#include <ChimeraTK/DummyBackend.h>

#include <boost/python/args.hpp>
#include <boost/python/numpy.hpp>

// no ; at line endings to be able to reuse in .def format
// any changes have to mirror the _userTypeExtensions dict in the python Device class
#define TEMPLATE_USERTYPE_POPULATION(FUNCTION_TEMPLATE, func_name)                                                     \
  FUNCTION_TEMPLATE(int8_t, func_name, _int8)                                                                          \
  FUNCTION_TEMPLATE(uint8_t, func_name, _uint8)                                                                        \
  FUNCTION_TEMPLATE(int16_t, func_name, _int16)                                                                        \
  FUNCTION_TEMPLATE(uint16_t, func_name, _uint16)                                                                      \
  FUNCTION_TEMPLATE(int32_t, func_name, _int32)                                                                        \
  FUNCTION_TEMPLATE(uint32_t, func_name, _uint32)                                                                      \
  FUNCTION_TEMPLATE(int64_t, func_name, _int64)                                                                        \
  FUNCTION_TEMPLATE(uint64_t, func_name, _uint64)                                                                      \
  FUNCTION_TEMPLATE(float, func_name, _float)                                                                          \
  FUNCTION_TEMPLATE(double, func_name, _double)                                                                        \
  FUNCTION_TEMPLATE(ChimeraTK::Boolean, func_name, _boolean)                                                           \
  FUNCTION_TEMPLATE(std::string, func_name, _string)

#define STRINGIFY(s) #s

#define TEMPLATECLASS_GET_GENERAL_TWODACCESSOR(userType, funcName, suffix)                                             \
  .def(STRINGIFY(funcName##suffix), mtca4upy::Device::getGeneralTwoDAccessor<userType>)

#define TEMPLATECLASS_GET_GENERAL_ONEDACCESSOR(userType, funcName, suffix)                                             \
  .def(STRINGIFY(funcName##suffix), mtca4upy::Device::getGeneralOneDAccessor<userType>)

#define TEMPLATECLASS_GET_GENERAL_SCALARACCESSOR(userType, funcName, suffix)                                           \
  .def(STRINGIFY(funcName##suffix), mtca4upy::Device::getGeneralScalarAccessor<userType>)

#define TEMPLATE_FILL_COMMON_REGISTER_FUNCS(accessorType, userType)                                                    \
  .def("isReadOnly", &accessorType<userType>::isReadOnly)                                                              \
      .def("isReadable", &accessorType<userType>::isReadable)                                                          \
      .def("isWriteable", &accessorType<userType>::isWriteable)                                                        \
      .def("isInitialised", &accessorType<userType>::isInitialised)                                                    \
      .def("getDescription", mtca4upy::GeneralRegisterAccessor<accessorType<userType>>::getDescription)                \
      .def("getUnit", mtca4upy::GeneralRegisterAccessor<accessorType<userType>>::getUnit)                              \
      .def("getName", mtca4upy::GeneralRegisterAccessor<accessorType<userType>>::getName)                              \
      .def("getVersionNumber", &accessorType<userType>::getVersionNumber)                                              \
      .def("setDataValidity", &accessorType<userType>::setDataValidity)                                                \
      .def("dataValidity", &accessorType<userType>::dataValidity)                                                      \
      .def("getId", &accessorType<userType>::getId)                                                                    \
      .def("getAccessModeFlagsString",                                                                                 \
          mtca4upy::GeneralRegisterAccessor<accessorType<userType>>::getAccessModeFlagsString)                         \
      .def("read", mtca4upy::GeneralRegisterAccessor<accessorType<userType>>::read)                                    \
      .def("readLatest", mtca4upy::GeneralRegisterAccessor<accessorType<userType>>::readLatest)                        \
      .def("readNonBlocking", mtca4upy::GeneralRegisterAccessor<accessorType<userType>>::readNonBlocking)              \
      .def("write", mtca4upy::GeneralRegisterAccessor<accessorType<userType>>::write)                                  \
      .def("writeDestructively", mtca4upy::GeneralRegisterAccessor<accessorType<userType>>::writeDestructively)

#define TEMPLATECLASS_TWODREGISTERACCESSOR(userType, className, class_suffix)                                          \
  bp::class_<ChimeraTK::TwoDRegisterAccessor<userType>>(STRINGIFY(className##class_suffix))                            \
      TEMPLATE_FILL_COMMON_REGISTER_FUNCS(ChimeraTK::TwoDRegisterAccessor, userType)                                   \
          .def("getNChannels", &ChimeraTK::TwoDRegisterAccessor<userType>::getNChannels)                               \
          .def("getNElementsPerChannel", &ChimeraTK::TwoDRegisterAccessor<userType>::getNElementsPerChannel);

#define TEMPLATECLASS_ONEDREGISTERACCESSOR(userType, className, class_suffix)                                          \
  bp::class_<ChimeraTK::OneDRegisterAccessor<userType>>(STRINGIFY(className##class_suffix))                            \
      TEMPLATE_FILL_COMMON_REGISTER_FUNCS(ChimeraTK::OneDRegisterAccessor, userType)                                   \
          .def("getNElements", &ChimeraTK::OneDRegisterAccessor<userType>::getNElements);

#define TEMPLATECLASS_SCALARREGISTERACCESSOR(userType, className, class_suffix)                                        \
  bp::class_<ChimeraTK::ScalarRegisterAccessor<userType>>(STRINGIFY(className##class_suffix))                          \
      TEMPLATE_FILL_COMMON_REGISTER_FUNCS(ChimeraTK::ScalarRegisterAccessor, userType)                                 \
          .def("readAndGet", &ChimeraTK::ScalarRegisterAccessor<userType>::readAndGet)                                 \
          .def("setAndWrite", &ChimeraTK::ScalarRegisterAccessor<userType>::setAndWrite)                               \
          .def("writeIfDifferent", &ChimeraTK::ScalarRegisterAccessor<userType>::writeIfDifferent);

namespace bp = boost::python;
namespace np = boost::python::numpy;

// This section defines function pointers used for overloading methods//
//****************************************************************************/

static boost::shared_ptr<ChimeraTK::Device> (*createDevice)(const std::string&) = &mtca4upy::createDevice;

//****************************************************************************//

// Auto-Overloading
BOOST_PYTHON_FUNCTION_OVERLOADS(open_overloads, mtca4upy::Device::open, 1, 2)

//****************************************************************************//

BOOST_PYTHON_MODULE(_da_python_bindings) {
  Py_Initialize();
  np::initialize();

  boost::python::to_python_converter<ChimeraTK::Boolean, CtkBoolean_to_python>();

  bool show_user_defined = true;
  bool show_signatures = false;
  bp::docstring_options doc_options(show_user_defined, show_signatures);

  bp::class_<ChimeraTK::Device>("Device")
      TEMPLATE_USERTYPE_POPULATION(TEMPLATECLASS_GET_GENERAL_TWODACCESSOR, getTwoDAccessor)
          TEMPLATE_USERTYPE_POPULATION(TEMPLATECLASS_GET_GENERAL_ONEDACCESSOR, getOneDAccessor)
              TEMPLATE_USERTYPE_POPULATION(TEMPLATECLASS_GET_GENERAL_SCALARACCESSOR, getScalarAccessor)
                  .def("getVoidAccessor", mtca4upy::Device::getVoidRegisterAccessor)
                  .def("getRegisterCatalogue", mtca4upy::Device::getRegisterCatalogue)
                  .def("activateAsyncRead", mtca4upy::Device::activateAsyncRead)
                  .def("getCatalogueMetadata", mtca4upy::Device::getCatalogueMetadata)
                  .def("open", (void (*)(ChimeraTK::Device&, std::string const&))0, open_overloads())
                  .def("read", mtca4upy::Device::read)
                  .def("write", mtca4upy::Device::write)
                  .def("close", mtca4upy::Device::close);

  TEMPLATE_USERTYPE_POPULATION(TEMPLATECLASS_SCALARREGISTERACCESSOR, ScalarAccessor)
  TEMPLATE_USERTYPE_POPULATION(TEMPLATECLASS_ONEDREGISTERACCESSOR, OneDAccessor)
  TEMPLATE_USERTYPE_POPULATION(TEMPLATECLASS_TWODREGISTERACCESSOR, TwoDAccessor)
  bp::class_<ChimeraTK::VoidRegisterAccessor>(
      "VoidRegisterAccessor", bp::init<boost::shared_ptr<ChimeraTK::NDRegisterAccessor<ChimeraTK::Void>>>())
      .def("isReadOnly", &ChimeraTK::VoidRegisterAccessor::isReadOnly)
      .def("isReadable", &ChimeraTK::VoidRegisterAccessor::isReadable)
      .def("isWriteable", &ChimeraTK::VoidRegisterAccessor::isWriteable)
      .def("isInitialised", &ChimeraTK::VoidRegisterAccessor::isInitialised)
      .def("getDescription", mtca4upy::GeneralRegisterAccessor<ChimeraTK::VoidRegisterAccessor>::getDescription)
      .def("getUnit", mtca4upy::GeneralRegisterAccessor<ChimeraTK::VoidRegisterAccessor>::getUnit)
      .def("getName", mtca4upy::GeneralRegisterAccessor<ChimeraTK::VoidRegisterAccessor>::getName)
      .def("getVersionNumber", &ChimeraTK::VoidRegisterAccessor::getVersionNumber)
      .def("setDataValidity", &ChimeraTK::VoidRegisterAccessor::setDataValidity)
      .def("dataValidity", &ChimeraTK::VoidRegisterAccessor::dataValidity)
      .def("getId", &ChimeraTK::VoidRegisterAccessor::getId)
      .def("getAccessModeFlagsString",
          mtca4upy::GeneralRegisterAccessor<ChimeraTK::VoidRegisterAccessor>::getAccessModeFlagsString)
      .def("read", mtca4upy::VoidRegisterAccessor::read)
      .def("readLatest", mtca4upy::VoidRegisterAccessor::readLatest)
      .def("readNonBlocking", mtca4upy::VoidRegisterAccessor::readNonBlocking)
      .def("writeDestructively", mtca4upy::VoidRegisterAccessor::writeDestructively)
      .def("write", mtca4upy::VoidRegisterAccessor::write);

  bp::def("createDevice", createDevice);
  bp::def("getDevice_no_alias", mtca4upy::getDevice_no_alias);
  bp::def("getDevice", mtca4upy::getDevice);
  bp::def("setDmapFile", mtca4upy::setDmapFile);
  bp::def("getDmapFile", mtca4upy::getDmapFile);

  bp::class_<ChimeraTK::RegisterCatalogue>("RegisterCatalogue", bp::init<ChimeraTK::RegisterCatalogue>())
      //.def("__iter__", bp::range(&ChimeraTK::RegisterCatalogue::begin, &ChimeraTK::RegisterCatalogue::end)) // TODO:
      // if someone needs to iterate through the register.
      // fix iteration implementation
      .def("hasRegister", mtca4upy::RegisterCatalogue::hasRegister)
      .def("getRegister", mtca4upy::RegisterCatalogue::getRegister);

  bp::class_<ChimeraTK::RegisterInfo>("RegisterInfo", bp::init<ChimeraTK::RegisterInfo>())
      .def("isReadable", &ChimeraTK::RegisterInfo::isReadable)
      .def("isValid", &ChimeraTK::RegisterInfo::isValid)
      .def("isWriteable", &ChimeraTK::RegisterInfo::isWriteable)
      .def("getRegisterName", mtca4upy::RegisterInfo::getRegisterName)
      .def("getSupportedAccessModes", mtca4upy::RegisterInfo::getSupportedAccessModes)
      .def("getNumberOfElements", &ChimeraTK::RegisterInfo::getNumberOfElements)
      .def("getNumberOfDimensions", &ChimeraTK::RegisterInfo::getNumberOfDimensions)
      .def("getNumberOfChannels", &ChimeraTK::RegisterInfo::getNumberOfChannels);

  bp::enum_<ChimeraTK::AccessMode>("AccessMode")
      .value("raw", ChimeraTK::AccessMode::raw)
      .value("wait_for_new_data", ChimeraTK::AccessMode::wait_for_new_data)
      .export_values();

  bp::enum_<ChimeraTK::DataValidity>("DataValidity")
      .value("ok", ChimeraTK::DataValidity::ok)
      .value("faulty", ChimeraTK::DataValidity::faulty)
      .export_values();

  bp::class_<ChimeraTK::TransferElementID>("TransferElementID")
      .def("isValid", &ChimeraTK::TransferElementID::isValid)
      .def("__ne__", &ChimeraTK::TransferElementID::operator!=)
      .def("__eq__", &ChimeraTK::TransferElementID::operator==);

  bp::class_<ChimeraTK::VersionNumber>("VersionNumber",
      "Class for generating and holding version numbers without exposing a numeric representation.\n"
      "\n"
      "Version numbers are used to resolve competing updates that are applied to the same process variable. For "
      "example, it they can help in breaking an infinite update loop that might occur when two process variables are "
      "related and update each other.\n"
      "\n"
      "They are also used to determine the order of updates made to different process variables.\n"
      "\n")
      .def("getTime", mtca4upy::VersionNumber::getTime, bp::args(""),
          "Return the time stamp associated with this version number. )\n"
          "\n")
      .def("__str__", &ChimeraTK::VersionNumber::operator std::string)
      .def("getNullVersion", mtca4upy::VersionNumber::getNullVersion)
      .def("__lt__", &ChimeraTK::VersionNumber::operator<)
      .def("__le__", &ChimeraTK::VersionNumber::operator<=)
      .def("__gt__", &ChimeraTK::VersionNumber::operator>)
      .def("__ge__", &ChimeraTK::VersionNumber::operator>=)
      .def("__ne__", &ChimeraTK::VersionNumber::operator!=)
      .def("__eq__", &ChimeraTK::VersionNumber::operator==);

  bp::register_ptr_to_python<boost::shared_ptr<ChimeraTK::Device>>();
  bp::register_ptr_to_python<boost::shared_ptr<ChimeraTK::VersionNumber>>();
  bp::register_ptr_to_python<boost::shared_ptr<ChimeraTK::TransferElementID>>();
  bp::register_ptr_to_python<boost::shared_ptr<ChimeraTK::RegisterCatalogue>>();
  bp::register_ptr_to_python<boost::shared_ptr<ChimeraTK::RegisterInfo>>();
}