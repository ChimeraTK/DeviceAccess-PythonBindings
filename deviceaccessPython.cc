#include "PythonModuleMethods.h"
#include <ChimeraTK/DummyBackend.h>
#include <boost/python/numpy.hpp>

namespace da = mtca4upy::DeviceAccess;
namespace oneD = mtca4upy::OneDAccessor;
namespace twoD = mtca4upy::TwoDAccessor;

// This section defines function pointers used for overloading methods//
//****************************************************************************//

static boost::shared_ptr<ChimeraTK::Device> (*createDevice)(const std::string&) = &mtca4upy::createDevice;

//****************************************************************************//

// Auto-Overloading
BOOST_PYTHON_FUNCTION_OVERLOADS(open_overloads, da::open, 1, 2)
BOOST_PYTHON_FUNCTION_OVERLOADS(getDevice_overloads, mtca4upy::getDevice, 0, 1)

//****************************************************************************//

// accessed through mtca4upy.py
// and not directly
BOOST_PYTHON_MODULE(_da_python_bindings) { // This module is

  // needed for the numpy ndarray c api to function correctly.
  Py_Initialize();
  //boost::python::numpy::initialize();
  _import_array();

  bp::class_<ChimeraTK::Device>("Device") // TODO: Find and change "Device" to a suitable name
      .def("get1DAccessor_int32", da::getOneDAccessor<int32_t>)
      .def("get1DAccessor_int64", da::getOneDAccessor<int64_t>)
      .def("get1DAccessor_float", da::getOneDAccessor<float>)
      .def("get1DAccessor_double", da::getOneDAccessor<double>)
      .def("getRaw1DAccessor", da::getRawOneDAccessor)
      .def("writeRaw", da::writeRaw)
      .def("get2DAccessor", da::getTwoDAccessor)
      .def("getTwoDAccessor_int32", da::getGeneralTwoDAccessor<int32_t>)
      .def("getCatalogueMetadata", da::getCatalogueMetadata)
      .def("open", (void (*)(ChimeraTK::Device&, std::string const&))0, open_overloads())
      .def("close", da::close);

  bp::class_<ChimeraTK::OneDRegisterAccessor<int32_t>>("OneDAccessor_int32")
      .def("read", oneD::read<int32_t>)
      .def("write", oneD::write<int32_t>)
      .def("getNumElements", oneD::getNumberOfElements<int32_t>);

  bp::class_<ChimeraTK::OneDRegisterAccessor<int64_t>>("OneDAccessor_int64")
      .def("read", oneD::read<int64_t>)
      .def("write", oneD::write<int64_t>)
      .def("getNumElements", oneD::getNumberOfElements<int64_t>);

  bp::class_<ChimeraTK::OneDRegisterAccessor<float>>("OneDAccessor_float")
      .def("read", oneD::read<float>)
      .def("write", oneD::write<float>)
      .def("getNumElements", oneD::getNumberOfElements<float>);

  bp::class_<ChimeraTK::OneDRegisterAccessor<double>>("OneDAccessor_double")
      .def("read", oneD::read<double>)
      .def("write", oneD::write<double>)
      .def("getNumElements", oneD::getNumberOfElements<double>);

  bp::class_<ChimeraTK::TwoDRegisterAccessor<double>>("TwoDAccessor_double")
      .def("read", twoD::read<double>)
      .def("getNChannels", twoD::getNChannels<double>)
      .def("getNElementsPerChannel", twoD::getNElementsPerChannel<double>);

  bp::class_<ChimeraTK::TwoDRegisterAccessor<int32_t>>("TwoDAccessor_int32")
      .def("read", twoD::read<int32_t>)
      .def("getNChannels", twoD::getNChannels<int32_t>)
      .def("getNElementsPerChannel", twoD::getNElementsPerChannel<int32_t>)
      .def("getBuffer", mtca4upy::TwoDRegisterAccessor::getBuffer<int32_t>);

  bp::def("createDevice", createDevice);
  // bp::def("getDevice", (boost::shared_ptr<ChimeraTK::Device>(*)(const std::string&))0, getDevice_overloads());
  bp::def("setDmapFile", mtca4upy::setDmapFile);
  bp::def("getDmapFile", mtca4upy::getDmapFile);
  bp::register_ptr_to_python<boost::shared_ptr<ChimeraTK::Device>>();

  bp::enum_<ChimeraTK::AccessMode>("AccessMode")
      .value("raw", ChimeraTK::AccessMode::raw)
      .value("wait_for_new_data", ChimeraTK::AccessMode::wait_for_new_data)
      .export_values();
}
