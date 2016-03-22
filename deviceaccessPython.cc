#include <mtca4u/DummyBackend.h>
#include <mtca4u/PcieBackend.h>
#include "PythonModuleMethods.h"

// This section defines function pointers used for overloading methods//
//****************************************************************************//

static boost::shared_ptr<mtca4u::Device>(*createDevice)(const std::string&,
                                                        const std::string&) =
    &mtca4upy::createDevice;
static boost::shared_ptr<mtca4u::Device>(*createDeviceFromCardAlias)(
    const std::string&) = &mtca4upy::createDevice;

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

  bp::class_<mtca4u::TwoDRegisterAccessor<float> >(
      "MuxDataAccessor")
      .def("getSequenceCount", mtca4upy::MuxDataAccessor::getSequenceCount)
      .def("getBlockCount", mtca4upy::MuxDataAccessor::getBlockCount)
      .def("populateArray", mtca4upy::MuxDataAccessor::copyReadInData);

  bp::def("createDevice", createDevice);
  bp::def("createDevice", createDeviceFromCardAlias);
  bp::def("setDmapFile", mtca4upy::setDmapFile);
  bp::register_ptr_to_python<boost::shared_ptr<mtca4u::Device> >();
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
}
