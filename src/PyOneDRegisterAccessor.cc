// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyOneDRegisterAccessor.h"

#include "HelperFunctions.h"

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/

  void PyOneDRegisterAccessor::set([[maybe_unused]] const UserTypeTemplateVariantNoVoid<Vector>& vec) {
    std::cout << "HIER PyOneDRegisterAccessor::set()" << std::endl;
  }

  /********************************************************************************************************************/

  void PyOneDRegisterAccessor::bind(py::module& m) {
    py::class_<PyOneDRegisterAccessor> arrayacc(m, "OneDRegisterAccessor");
    arrayacc
        .def(py::init<>())

        .def("set", &PyOneDRegisterAccessor::set, "Set the values of the array of UserType.", py::arg("newValue"));
  }

  /********************************************************************************************************************/

} // namespace ChimeraTK
