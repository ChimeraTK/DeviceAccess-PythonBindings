#include <mtca4u/DummyBackend.h>
#include <mtca4u/PcieBackend.h>
#include "PythonModuleMethods.h"
#include "MultiplexedDataAccessorWrapper.h"

// This section defines function pointers used for overloading methods//
//****************************************************************************//

static boost::shared_ptr<mtca4u::Device>(*createDevice)(const std::string&, const std::string&) = &mtca4upy::createDevice;
static boost::shared_ptr<mtca4u::Device>(*createDeviceFromCardAlias)( const std::string&) = &mtca4upy::createDevice;

//****************************************************************************//

BOOST_PYTHON_MODULE(mtca4udeviceaccess) { // This module is
  // actually accessed through mtca4upy.py
  // and not directly
  bp::class_<mtca4u::Device>(
      "Device") // TODO: Find and change "Device" to a suitable name
      .def("writeRaw", mtca4upy::DeviceAccess::writeRaw)
      .def("getRegisterAccessor", mtca4upy::DeviceAccess::getRegisterAccessor)
      .def("getMultiplexedDataAccessor",
           mtca4upy::DeviceAccess::getMultiplexedDataAccessor);

  bp::class_<mtca4u::Device::RegisterAccessor,
             boost::shared_ptr<mtca4u::Device::RegisterAccessor>,
             boost::noncopyable>("RegisterAccessor", bp::no_init)
      .def("read", mtca4upy::RegisterAccessor::read)
      .def("write", mtca4upy::RegisterAccessor::write)
      .def("readRaw", mtca4upy::RegisterAccessor::readRaw)
      .def("writeRaw", mtca4upy::RegisterAccessor::writeRaw)
      .def("readDMARaw", mtca4upy::RegisterAccessor::readDMARaw)
      .def("getNumElements", mtca4upy::RegisterAccessor::size);

  /*
  * define the multipleddataaccessor class in the module. here we specify its
  * python interface in the boost generated python module.
  *
  * The MultiplexedDataAccessor is an abstract class. We would also need to
  * register a shared pointer object to this type (because the factory we use
  * to create it gives back a shared pointer). What we have below should take
  * care of the abstract class part
  */
  bp::class_<mtca4upy::MultiplexedDataAccessorWrapper, boost::noncopyable>(
      "MuxDataAccessor",
      bp::init<const boost::shared_ptr<mtca4u::DeviceBackend>&,
               const std::vector<mtca4u::FixedPointConverter>&>())
      .def("readFromDevice", mtca4upy::MuxDataAccessor::readInDataFromCard)
      .def("getSequenceCount", mtca4upy::MuxDataAccessor::getSequenceCount)
      .def("getBlockCount", mtca4upy::MuxDataAccessor::getBlockCount)
      .def("populateArray", mtca4upy::MuxDataAccessor::copyReadInData);

  bp::def("createDevice", createDevice);
  bp::def("createDevice", createDeviceFromCardAlias);
  bp::def("setDmapFile", mtca4upy::setDmapFile);
  bp::register_ptr_to_python<boost::shared_ptr<mtca4u::Device> >();
  bp::register_ptr_to_python<
      boost::shared_ptr<mtca4u::MultiplexedDataAccessor<float> > >();
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
}
