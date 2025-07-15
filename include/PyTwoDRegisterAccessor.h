// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <pybind11/pybind11.h>
// pybind11.h must come first

#include "PyTransferElement.h"

#include <ChimeraTK/AccessMode.h>
#include <ChimeraTK/TwoDRegisterAccessor.h>
#include <ChimeraTK/VariantUserTypes.h>

#include <pybind11/numpy.h>

#include <iostream>

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/

  class PyTwoDRegisterAccessor : public PyTransferElement<PyTwoDRegisterAccessor> {
   public:
    PyTwoDRegisterAccessor() : _accessor(TwoDRegisterAccessor<int>()), _continuousBuffer(std::vector<int>()) {}
    PyTwoDRegisterAccessor(PyTwoDRegisterAccessor&&) = default;
    ~PyTwoDRegisterAccessor();

    template<typename UserType>
    explicit PyTwoDRegisterAccessor(ChimeraTK::TwoDRegisterAccessor<UserType> acc)
    : _accessor(acc), _continuousBuffer(std::vector<UserType>()) {}

    // UserTypeTemplateVariantNoVoid expects a single template argument, std::vector has multiple (with defaults)...
    template<typename T>
    using Vector = std::vector<T>;

    template<typename T>
    using VVector = std::vector<std::vector<T>>;

    void read();
    void readLatest();
    void readNonBlocking();

    size_t getNChannels();
    size_t getNElementsPerChannel();

    void set(const UserTypeTemplateVariantNoVoid<VVector>& vec);

    py::object get() const;

    std::string repr(py::object& acc) const;

    py::buffer_info getBufferInfo() const;

    py::object getattr(const std::string& name) const {
      std::cout << "getattr with " << name << " called for TwoDRegisterAccessor" << std::endl;
      return get().attr(name.c_str());
    }
    py::object getitem(size_t index) const;

    static void bind(py::module& mod);

    mutable UserTypeTemplateVariantNoVoid<TwoDRegisterAccessor> _accessor;

   private:
    mutable UserTypeTemplateVariantNoVoid<Vector> _continuousBuffer;
    void copyToBuffer();
  };

  /********************************************************************************************************************/

} // namespace ChimeraTK