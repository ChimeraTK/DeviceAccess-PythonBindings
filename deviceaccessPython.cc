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
  .def(STRINGIFY(funcName##suffix), da::getGeneralTwoDAccessor<userType>)

#define TEMPLATECLASS_GET_GENERAL_ONEDACCESSOR(userType, funcName, suffix)                                             \
  .def(STRINGIFY(funcName##suffix), da::getGeneralOneDAccessor<userType>)

#define TEMPLATECLASS_GET_GENERAL_SCALARACCESSOR(userType, funcName, suffix)                                           \
  .def(STRINGIFY(funcName##suffix), da::getGeneralScalarAccessor<userType>)

#define TEMPLATE_FILL_COMMON_REGISTER_FUNCS(accessorType, userType)                                                    \
  .def("isReadOnly", mtca4upy::GeneralRegisterAccessor::isReadOnly<ChimeraTK::accessorType<userType>>)                 \
      .def("isReadable", mtca4upy::GeneralRegisterAccessor::isReadable<ChimeraTK::accessorType<userType>>)             \
      .def("isWriteable", mtca4upy::GeneralRegisterAccessor::isWriteable<ChimeraTK::accessorType<userType>>)           \
      .def("isInitialised", mtca4upy::GeneralRegisterAccessor::isInitialised<ChimeraTK::accessorType<userType>>)       \
      .def("getDescription", mtca4upy::GeneralRegisterAccessor::getDescription<ChimeraTK::accessorType<userType>>)     \
      .def("getVersionNumber", mtca4upy::GeneralRegisterAccessor::getVersionNumber<ChimeraTK::accessorType<userType>>) \
      .def("setDataValidity", mtca4upy::GeneralRegisterAccessor::setDataValidity<ChimeraTK::accessorType<userType>>)   \
      .def("dataValidity", mtca4upy::GeneralRegisterAccessor::dataValidity<ChimeraTK::accessorType<userType>>)         \
      .def("getUnit", mtca4upy::GeneralRegisterAccessor::getUnit<ChimeraTK::accessorType<userType>>)                   \
      .def("getName", mtca4upy::GeneralRegisterAccessor::getName<ChimeraTK::accessorType<userType>>)                   \
      .def("getId", mtca4upy::GeneralRegisterAccessor::getId<ChimeraTK::accessorType<userType>>)                       \
      .def("getAccessModeFlagsString",                                                                                 \
          mtca4upy::GeneralRegisterAccessor::getAccessModeFlagsString<ChimeraTK::accessorType<userType>>)              \
      .def("read", mtca4upy::GeneralRegisterAccessor::read<ChimeraTK::accessorType<userType>>)                         \
      .def("readLatest", mtca4upy::GeneralRegisterAccessor::readLatest<ChimeraTK::accessorType<userType>>)             \
      .def("readNonBlocking", mtca4upy::GeneralRegisterAccessor::readNonBlocking<ChimeraTK::accessorType<userType>>)   \
      .def("write", mtca4upy::GeneralRegisterAccessor::write<ChimeraTK::accessorType<userType>>)                       \
      .def("writeDestructively",                                                                                       \
          mtca4upy::GeneralRegisterAccessor::writeDestructively<ChimeraTK::accessorType<userType>>)

#define TEMPLATECLASS_TWODREGISTERACCESSOR(userType, className, class_suffix)                                          \
  bp::class_<ChimeraTK::TwoDRegisterAccessor<userType>>(STRINGIFY(className##class_suffix))                            \
      TEMPLATE_FILL_COMMON_REGISTER_FUNCS(TwoDRegisterAccessor, userType)                                              \
          .def("getNChannels", mtca4upy::TwoDRegisterAccessor::getNChannels<userType>)                                 \
          .def("getNElementsPerChannel", mtca4upy::TwoDRegisterAccessor::getNElementsPerChannel<userType>);

#define TEMPLATECLASS_ONEDREGISTERACCESSOR(userType, className, class_suffix)                                          \
  bp::class_<ChimeraTK::OneDRegisterAccessor<userType>>(STRINGIFY(className##class_suffix))                            \
      TEMPLATE_FILL_COMMON_REGISTER_FUNCS(OneDRegisterAccessor, userType)                                              \
          .def("getNElements", mtca4upy::OneDRegisterAccessor::getNElements<userType>);

#define TEMPLATECLASS_SCALARREGISTERACCESSOR(userType, className, class_suffix)                                        \
  bp::class_<ChimeraTK::ScalarRegisterAccessor<userType>>(STRINGIFY(className##class_suffix))                          \
      TEMPLATE_FILL_COMMON_REGISTER_FUNCS(ScalarRegisterAccessor, userType)                                            \
          .def("readAndGet", mtca4upy::ScalarRegisterAccessor::readAndGet<userType>)                                   \
          .def("setAndWrite", mtca4upy::ScalarRegisterAccessor::setAndWrite<userType>)                                 \
          .def("writeIfDifferent", mtca4upy::ScalarRegisterAccessor::writeIfDifferent<userType>);

namespace bp = boost::python;
namespace np = boost::python::numpy;

namespace da = mtca4upy::DeviceAccess;

// This section defines function pointers used for overloading methods//
//****************************************************************************/

static boost::shared_ptr<ChimeraTK::Device> (*createDevice)(const std::string&) = &mtca4upy::createDevice;

//****************************************************************************//

// Auto-Overloading
BOOST_PYTHON_FUNCTION_OVERLOADS(open_overloads, da::open, 1, 2)

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
                  .def("getVoidAccessor", da::getVoidRegisterAccessor)
                  .def("getRegisterCatalogue", da::getRegisterCatalogue)
                  .def("activateAsyncRead", da::activateAsyncRead)
                  .def("getCatalogueMetadata", da::getCatalogueMetadata)
                  .def("open", (void (*)(ChimeraTK::Device&, std::string const&))0, open_overloads())
                  .def("read", da::read)
                  .def("write", da::write)
                  .def("close", da::close);

  TEMPLATE_USERTYPE_POPULATION(TEMPLATECLASS_SCALARREGISTERACCESSOR, ScalarAccessor)
  TEMPLATE_USERTYPE_POPULATION(TEMPLATECLASS_ONEDREGISTERACCESSOR, OneDAccessor)
  TEMPLATE_USERTYPE_POPULATION(TEMPLATECLASS_TWODREGISTERACCESSOR, TwoDAccessor)
  bp::class_<ChimeraTK::VoidRegisterAccessor>(
      "VoidRegisterAccessor", bp::init<boost::shared_ptr<ChimeraTK::NDRegisterAccessor<ChimeraTK::Void>>>())
      .def("isReadOnly", mtca4upy::GeneralRegisterAccessor::isReadOnly<ChimeraTK::VoidRegisterAccessor>)
      .def("isReadable", mtca4upy::GeneralRegisterAccessor::isReadable<ChimeraTK::VoidRegisterAccessor>)
      .def("isWriteable", mtca4upy::GeneralRegisterAccessor::isWriteable<ChimeraTK::VoidRegisterAccessor>)
      .def("isInitialised", mtca4upy::GeneralRegisterAccessor::isInitialised<ChimeraTK::VoidRegisterAccessor>)
      .def("getDescription", mtca4upy::GeneralRegisterAccessor::getDescription<ChimeraTK::VoidRegisterAccessor>)
      .def("getVersionNumber", mtca4upy::GeneralRegisterAccessor::getVersionNumber<ChimeraTK::VoidRegisterAccessor>)
      .def("setDataValidity", mtca4upy::GeneralRegisterAccessor::setDataValidity<ChimeraTK::VoidRegisterAccessor>)
      .def("dataValidity", mtca4upy::GeneralRegisterAccessor::dataValidity<ChimeraTK::VoidRegisterAccessor>)
      .def("getUnit", mtca4upy::GeneralRegisterAccessor::getUnit<ChimeraTK::VoidRegisterAccessor>)
      .def("getName", mtca4upy::GeneralRegisterAccessor::getName<ChimeraTK::VoidRegisterAccessor>)
      .def("getId", mtca4upy::GeneralRegisterAccessor::getId<ChimeraTK::VoidRegisterAccessor>)
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
      .def("isReadable", mtca4upy::RegisterInfo::isReadable)
      .def("isValid", mtca4upy::RegisterInfo::isValid)
      .def("isWriteable", mtca4upy::RegisterInfo::isWriteable)
      .def("getRegisterName", mtca4upy::RegisterInfo::getRegisterName)
      .def("getSupportedAccessModes", mtca4upy::RegisterInfo::getSupportedAccessModes)
      .def("getNumberOfElements", mtca4upy::RegisterInfo::getNumberOfElements)
      .def("getNumberOfDimensions", mtca4upy::RegisterInfo::getNumberOfDimensions)
      .def("getNumberOfChannels", mtca4upy::RegisterInfo::getNumberOfChannels);

  bp::enum_<ChimeraTK::AccessMode>("AccessMode")
      .value("raw", ChimeraTK::AccessMode::raw)
      .value("wait_for_new_data", ChimeraTK::AccessMode::wait_for_new_data)
      .export_values();

  bp::enum_<ChimeraTK::DataValidity>("DataValidity")
      .value("ok", ChimeraTK::DataValidity::ok)
      .value("faulty", ChimeraTK::DataValidity::faulty)
      .export_values();

  bp::class_<ChimeraTK::TransferElementID>("TransferElementID")
      .def("isValid", mtca4upy::TransferElementID::isValid)
      .def("__lt__", mtca4upy::TransferElementID::lt)
      .def("__le__", mtca4upy::TransferElementID::le)
      .def("__gt__", mtca4upy::TransferElementID::gt)
      .def("__ge__", mtca4upy::TransferElementID::ge)
      .def("__ne__", mtca4upy::TransferElementID::ne)
      .def("__eq__", mtca4upy::TransferElementID::eq);

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
      .def("__str__", mtca4upy::VersionNumber::str)
      .def("getNullVersion", mtca4upy::VersionNumber::getNullVersion)
      .def("__lt__", mtca4upy::VersionNumber::lt)
      .def("__le__", mtca4upy::VersionNumber::le)
      .def("__gt__", mtca4upy::VersionNumber::gt)
      .def("__ge__", mtca4upy::VersionNumber::ge)
      .def("__ne__", mtca4upy::VersionNumber::ne)
      .def("__eq__", mtca4upy::VersionNumber::eq);

  bp::register_ptr_to_python<boost::shared_ptr<ChimeraTK::Device>>();
  bp::register_ptr_to_python<boost::shared_ptr<ChimeraTK::VersionNumber>>();
  bp::register_ptr_to_python<boost::shared_ptr<ChimeraTK::TransferElementID>>();
  bp::register_ptr_to_python<boost::shared_ptr<ChimeraTK::RegisterCatalogue>>();
  bp::register_ptr_to_python<boost::shared_ptr<ChimeraTK::RegisterInfo>>();
}