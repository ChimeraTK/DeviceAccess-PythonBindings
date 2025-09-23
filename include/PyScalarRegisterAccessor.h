// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <pybind11/pybind11.h>
// pybind11.h must come first

#include "PyTransferElement.h"
#include "PyVersionNumber.h"

#include <ChimeraTK/AccessMode.h>
#include <ChimeraTK/ScalarRegisterAccessor.h>
#include <ChimeraTK/VariantUserTypes.h>

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/

  class PyScalarRegisterAccessor : public PyTransferElement<PyScalarRegisterAccessor> {
   public:
    PyScalarRegisterAccessor() : _accessor(ScalarRegisterAccessor<int>()) {}
    PyScalarRegisterAccessor(PyScalarRegisterAccessor&&) = default;
    ~PyScalarRegisterAccessor();

    template<typename UserType>
    explicit PyScalarRegisterAccessor(ChimeraTK::ScalarRegisterAccessor<UserType> acc) : _accessor(acc) {}

    py::object readAndGet();

    void setAndWrite(const UserTypeVariantNoVoid& val, const PyVersionNumber& versionNumber);
    void setAndWriteArray(const py::array& val, const PyVersionNumber& versionNumber);

    void writeIfDifferent(const UserTypeVariantNoVoid& val, const PyVersionNumber& versionNumber);
    void writeIfDifferentArray(const py::array& val, const PyVersionNumber& versionNumber);

    void set(const UserTypeVariantNoVoid& val);
    void setArray(const py::array& val);
    void setList(const py::list& val);

    py::object get() const;

    std::string repr(py::object& acc) const;

    py::object getattr(const std::string& name) const { return get().attr(name.c_str()); }

    static void bind(py::module& mod);

    // ScalarRegisterAccessor has 2 template parameters (2nd has default), but UserTypeTemplateVariantNoVoid expects a
    // template template parameter with a single parameter. gcc seems to compile it anyway, but clang does not.
    template<typename T>
    using ScalarRegisterAccessorT = ScalarRegisterAccessor<T>;

   private:
    mutable UserTypeTemplateVariantNoVoid<ScalarRegisterAccessorT> _accessor;
    friend PyTransferElement;
  };

  /********************************************************************************************************************/

} // namespace ChimeraTK
