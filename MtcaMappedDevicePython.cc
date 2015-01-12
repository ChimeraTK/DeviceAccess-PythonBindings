#include <MtcaMappedDevice/devPCIE.h>
#include "WrapperMethods.h"

namespace bp = boost::python;;

// This covers the version of openDev method with default arguments
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(openDevDefaultArgs,
                                       mtca4u::devPCIE::openDev, 1, 3)

BOOST_PYTHON_MODULE(mtcamappeddevice) {
  bp::class_<mtca4u::devConfigBase>("devConfigBase");

  bp::class_<mtca4u::devPCIE, boost::noncopyable/*copy constructor not defined*/>("devPCIE")
      .def("openDev", &mtca4u::devPCIE::openDev, openDevDefaultArgs())
      .def("closeDev", &mtca4u::devPCIE::closeDev)
      .def("readReg", &readReg) // TODO: assign a proper namespace for code in WrapperMethods.cc
      .def("writeReg", &mtca4u::devPCIE::writeReg)
      .def("readDMA", &readDMA)
      .def("readArea", &readArea)
      .def("writeArea", &writeArea)
      .def("readDeviceInfo", &readDeviceInfo);
}



