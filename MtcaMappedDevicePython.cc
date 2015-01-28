#include <MtcaMappedDevice/devPCIE.h>
#include "WrapperMethods.h"
#include "DevBaseWrapper.h"

namespace bp = boost::python;

// This covers the version of openDev method with default arguments
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(openDevDefaultArgs,
                                       mtca4u::devPCIE::openDev, 1, 3)

/*BOOST_PYTHON_MODULE(mtcamappeddevice) {
  bp::class_<mtca4u::devConfigBase>("devConfigBase");

  bp::class_<mtca4u::devPCIE,
             boost::noncopyable copy constructor not defined>("devPCIE")
      .def("openDev", &mtca4u::devPCIE::openDev, openDevDefaultArgs())
      .def("closeDev", &mtca4u::devPCIE::closeDev)
      .def("readReg", &readReg) // TODO: assign a proper namespace for code in
                                // WrapperMethods.cc
      .def("writeReg", &mtca4u::devPCIE::writeReg)
      .def("readDMA", &readDMA)
      .def("readArea", &readArea)
      .def("writeArea", &writeArea)
      .def("readDeviceInfo", &readDeviceInfo);
}*/

BOOST_PYTHON_MODULE(mtcamappeddevice) {
  bp::class_<mtca4upy::DevBaseWrapper,
             boost::noncopyable /*copy constructor not defined*/>("devPCIE")
      .def("open", bp::pure_virtual(&mtca4upy::openDev))
      .def("closeDev", &mtca4u::devBase::closeDev)
      .def("readReg",
           &mtca4upy::readReg) // TODO: assign a proper namespace for code in
                               // WrapperMethods.cc
      .def("writeReg", &mtca4u::devBase::writeReg)
      .def("readDMA", &mtca4upy::readDMA)
      .def("readArea", &mtca4upy::readArea)
      .def("writeArea", &mtca4upy::writeArea)
      .def("readDeviceInfo", &mtca4upy::readDeviceInfo);
}
