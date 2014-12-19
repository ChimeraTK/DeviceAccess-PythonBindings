#include <boost/python.hpp>
#include <MtcaMappedDevice/devPCIE.h>

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(openDevDefaultArgs,
                                       mtca4u::devPCIE::openDev, 1, 3)

BOOST_PYTHON_MODULE(mtcamappeddevice) {
  boost::python::class_<mtca4u::devConfigBase>("devConfigBase");

  boost::python::class_<mtca4u::devPCIE, boost::noncopyable>("devPCIE")
      .def("openDev", &mtca4u::devPCIE::openDev, openDevDefaultArgs())
      .def("closeDev", &mtca4u::devPCIE::closeDev)
      .def("readReg", &mtca4u::devPCIE::readReg)
      .def("writeReg", &mtca4u::devPCIE::writeReg)
      .def("readArea", &mtca4u::devPCIE::readArea)
      .def("writeArea", &mtca4u::devPCIE::writeArea)
      .def("readDMA", &mtca4u::devPCIE::readDMA)
      .def("writeDMA", &mtca4u::devPCIE::writeDMA)
      .def("readDeviceInfo", &mtca4u::devPCIE::readDeviceInfo);
}
