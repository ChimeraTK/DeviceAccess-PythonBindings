#include <MtcaMappedDevice/devPCIE.h>
#include "WrapperMethods.h"
#include "DevBaseWrapper.h"
#include "SimpleFactory.h"

namespace bp = boost::python;

// This covers the version of openDev method with default arguments
/*BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(openDevDefaultArgs,
                                       mtca4u::devPCIE::openDev, 1, 3)*/ // TODO: remove
                                                             // this

BOOST_PYTHON_MODULE(mtcamappeddevice) {
  bp::class_<mtca4upy::DevBaseWrapper,
             boost::noncopyable /*copy constructor not defined*/>("devBase")
      .def("openDev", bp::pure_virtual(&mtca4upy::openDev))
      .def("closeDev", bp::pure_virtual(&mtca4upy::closeDev))
      .def("readReg", bp::pure_virtual(&mtca4upy::readReg))
      .def("writeReg", bp::pure_virtual(&mtca4upy::writeReg))
      .def("readDMA", bp::pure_virtual(&mtca4upy::readDMA))
      .def("readArea", bp::pure_virtual(&mtca4upy::readArea))
      .def("writeArea", bp::pure_virtual(&mtca4upy::writeArea))
      .def("readDeviceInfo", bp::pure_virtual(&mtca4upy::readDeviceInfo));

  bp::class_<mtca4upy::Device, boost::noncopyable>("Device")
      .def("createPCIEDevice", &mtca4upy::Device::createPCIEDevice)
      .def("createDummyDevice", &mtca4upy::Device::createDummyDevice);

  bp::register_ptr_to_python<boost::shared_ptr<mtca4u::devBase> >();
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
}
