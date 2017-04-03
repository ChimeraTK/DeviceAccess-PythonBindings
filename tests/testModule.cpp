/*
 * testHelperFunctions.cpp
 *
 *  Created on: Apr 3, 2017
 *      Author: varghese
 */

#include "HelperFunctions.h"
#include <boost/python.hpp>

namespace bp = boost::python;

BOOST_PYTHON_MODULE(testmodule) {

  bp::def("extractDataType", mtca4upy::extractDataType);

  bp::enum_<mtca4upy::numpyDataTypes>("numpyDataTypes")
      .value("INT32", mtca4upy::INT32)
      .value("INT64", mtca4upy::INT64)
      .value("FLOAT32", mtca4upy::FLOAT32)
      .value("FLOAT64", mtca4upy::FLOAT64)
      .value("USUPPORTED_TYPE", mtca4upy::USUPPORTED_TYPE);

  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
}
