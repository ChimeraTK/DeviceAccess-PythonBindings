#include <boost/python.hpp>
#include <MtcaMappedDevice/devPCIE.h>

BOOST_PYTHON_MODULE(mtcamappeddevice) {
  boost::python::class_<mtca4u::devPCIE, boost::noncopyable>("devPCIE")
      .def("openDev", &mtca4u::devPCIE::openDev);
}
