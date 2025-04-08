// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyTransferElement.h"

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/

  void PyTransferElementBase::bind(py::module& mod) {
    py::class_<PyTransferElementBase>(mod, "PyTransferElementBase");
  }

  /********************************************************************************************************************/

  const std::set<std::string> PyTransferElementBase::specialFunctionsToEmulateNumeric{"__eq__", "__ne__", "__lt__",
      "__le__", "__gt__", "__ge__", "__add__", "__sub__", "__mul__", "__matmul__", "__truediv__", "__floordiv__",
      "__mod__", "__divmod__", "__pow__", "__lshift__", "__rshift__", "__and__", "__xor__", "__or__", "__radd__",
      "__rsub__", "__rmul__", "__rmatmul__", "__rtruediv__", "__rfloordiv__", "__rmod__", "__rdivmod__", "__rpow__",
      "__rlshift__", "__rrshift__", "__rand__", "__rxor__", "__ror__", "__round__"};

  /********************************************************************************************************************/

  const std::set<std::string> PyTransferElementBase::specialUnaryFunctionsToEmulateNumeric{"__neg__", "__pos__",
      "__abs__", "__invert__", "__int__", "__float__", "__round__", "__trunc__", "__floor__", "__ceil__", "__str__",
      "__bool__"};

  /********************************************************************************************************************/

} // namespace ChimeraTK