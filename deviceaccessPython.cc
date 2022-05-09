#include "PythonModuleMethods.h"
#include <ChimeraTK/DummyBackend.h>
#include <boost/python/numpy.hpp>

namespace bp = boost::python;
namespace np = boost::python::numpy;

namespace da = mtca4upy::DeviceAccess;

// This section defines function pointers used for overloading methods//
//****************************************************************************//

static boost::shared_ptr<ChimeraTK::Device> (*createDevice)(const std::string&) = &mtca4upy::createDevice;

//****************************************************************************//

// Auto-Overloading
BOOST_PYTHON_FUNCTION_OVERLOADS(open_overloads, da::open, 1, 2)

//****************************************************************************//

// accessed through mtca4upy.py
// and not directly
BOOST_PYTHON_MODULE(discarded) { // This module is

  // needed for the numpy ndarray c api to function correctly.
  //Py_Initialize();
  //boost::python::numpy::initialize();

  bp::register_ptr_to_python<boost::shared_ptr<ChimeraTK::Device>>();
}

BOOST_PYTHON_MODULE(_da_python_bindings) {
  Py_Initialize();
  np::initialize();

  bp::class_<ChimeraTK::Device>("Device") // TODO: Find and change "Device" to a suitable name
      .def("getTwoDAccessor_int32", da::getGeneralTwoDAccessor<int32_t>)
      .def("getCatalogueMetadata", da::getCatalogueMetadata)
      .def("open", (void (*)(ChimeraTK::Device&, std::string const&))0, open_overloads())
      .def("close", da::close);

  bp::class_<ChimeraTK::TwoDRegisterAccessor<int32_t>>("TwoDAccessor_int32")
      .def("read", mtca4upy::TwoDRegisterAccessor::read<int32_t>)
      .def("readLatest", mtca4upy::TwoDRegisterAccessor::readLatest<int32_t>)
      .def("readNonBlocking", mtca4upy::TwoDRegisterAccessor::readNonBlocking<int32_t>)
      .def("write", mtca4upy::TwoDRegisterAccessor::write<int32_t>)
      .def("writeDestructively", mtca4upy::TwoDRegisterAccessor::writeDestructively<int32_t>)
      .def("getNChannels", mtca4upy::TwoDRegisterAccessor::getNChannels<int32_t>)
      .def("getNElementsPerChannel", mtca4upy::TwoDRegisterAccessor::getNElementsPerChannel<int32_t>)
      .def("isReadOnly", mtca4upy::TwoDRegisterAccessor::isReadOnly<int32_t>)
      .def("isReadable", mtca4upy::TwoDRegisterAccessor::isReadable<int32_t>)
      .def("isWriteable", mtca4upy::TwoDRegisterAccessor::isWriteable<int32_t>)
      .def("isInitialised", mtca4upy::TwoDRegisterAccessor::isInitialised<int32_t>)
      .def("getUnit", mtca4upy::TwoDRegisterAccessor::getUnit<int32_t>)
      .def("getDescription", mtca4upy::TwoDRegisterAccessor::getDescription<int32_t>)
      .def("getName", mtca4upy::TwoDRegisterAccessor::getName<int32_t>);

  bp::def("createDevice", createDevice);
  bp::def("getDevice_no_alias", mtca4upy::getDevice_no_alias);
  bp::def("getDevice", mtca4upy::getDevice);
  bp::def("setDmapFile", mtca4upy::setDmapFile);
  bp::def("getDmapFile", mtca4upy::getDmapFile);

  bp::enum_<ChimeraTK::AccessMode>("AccessMode")
      .value("raw", ChimeraTK::AccessMode::raw)
      .value("wait_for_new_data", ChimeraTK::AccessMode::wait_for_new_data)
      .export_values();

  bp::register_ptr_to_python<boost::shared_ptr<ChimeraTK::Device>>();
}