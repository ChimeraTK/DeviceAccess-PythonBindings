// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <pybind11/pybind11.h>
// pybind11.h must come first

#include "PyTransferElement.h"

#include <ChimeraTK/AccessMode.h>
#include <ChimeraTK/OneDRegisterAccessor.h>
#include <ChimeraTK/VariantUserTypes.h>

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/

  class PyOneDRegisterAccessor : public PyTransferElement<PyOneDRegisterAccessor> {
   public:
    PyOneDRegisterAccessor() : _accessor(OneDRegisterAccessor<int>()) {}
    PyOneDRegisterAccessor(PyOneDRegisterAccessor&&) = default;
    ~PyOneDRegisterAccessor();

    template<typename UserType>
    explicit PyOneDRegisterAccessor(ChimeraTK::OneDRegisterAccessor<UserType> acc) : _accessor(acc) {}

    // UserTypeTemplateVariantNoVoid expects a single template argument, std::vector has multiple (with defaults)...
    template<typename T>
    using Vector = std::vector<T>;

    py::object readAndGet();

    void setAndWrite(const UserTypeTemplateVariantNoVoid<Vector>& vec, const PyVersionNumber& versionNumber);

    size_t getNElements() const;

    void set(const UserTypeTemplateVariantNoVoid<Vector>& vec);

    py::object get() const;

    py::object getitem(size_t index) const;

    void setitem(size_t index, const UserTypeVariantNoVoid& val);

    UserTypeVariantNoVoid getAsCooked(uint element);
    void setAsCooked(uint element, UserTypeVariantNoVoid value);

    std::string repr(py::object& acc) const;

    py::buffer_info getBufferInfo();

    py::object getattr(const std::string& name) const { return get().attr(name.c_str()); }

    static void bind(py::module& mod);

   private:
    mutable UserTypeTemplateVariantNoVoid<OneDRegisterAccessor> _accessor;
    friend PyTransferElement;
  };

  /********************************************************************************************************************/

} // namespace ChimeraTK
