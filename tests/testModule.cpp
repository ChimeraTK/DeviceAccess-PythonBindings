#include "HelperFunctions.h"

#include <boost/python.hpp>

namespace bp = boost::python;

void testNumpyObjManager(mtca4upy::NumpyObject&);

BOOST_PYTHON_MODULE(testmodule) {
  // needed for the numpy ndarray c api to function correctly.
  Py_Initialize();
  _import_array();

  bp::def("extractDataType", mtca4upy::extractDataType);

  bp::enum_<mtca4upy::numpyDataTypes>("numpyDataTypes")
      .value("INT32", mtca4upy::INT32)
      .value("INT64", mtca4upy::INT64)
      .value("FLOAT32", mtca4upy::FLOAT32)
      .value("FLOAT64", mtca4upy::FLOAT64)
      .value("USUPPORTED_TYPE", mtca4upy::USUPPORTED_TYPE);

  bp::def("testNumpyObjManager", testNumpyObjManager);
}

// Here just to check the binding
void testNumpyObjManager(mtca4upy::NumpyObject& /*numpyArray*/) {}
