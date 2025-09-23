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

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/

  class PyTwoDRegisterAccessor : public PyTransferElement<PyTwoDRegisterAccessor> {
   public:
    PyTwoDRegisterAccessor() : _accessor(TwoDRegisterAccessor<int>()), _continuousBuffer(std::vector<int>()) {}
    PyTwoDRegisterAccessor(PyTwoDRegisterAccessor&&) = default;
    ~PyTwoDRegisterAccessor();

    // UserTypeTemplateVariantNoVoid expects a single template argument, std::vector has multiple (with defaults)...
    template<typename T>
    using Vector = std::vector<T>;

    template<typename T>
    using VVector = std::vector<std::vector<T>>;

    void read();
    bool readLatest();
    bool readNonBlocking();
    void write(const ChimeraTK::VersionNumber& versionNumber = ChimeraTK::VersionNumber{});
    void writeDestructively(const ChimeraTK::VersionNumber& versionNumber = ChimeraTK::VersionNumber{});

    size_t getNChannels();
    size_t getNElementsPerChannel();

    void set(const UserTypeTemplateVariantNoVoid<VVector>& vec);

    template<typename AccessorType>
    void setTE(AccessorType incomingAcc) {
      PyTransferElement::setTE(incomingAcc);

      std::vector<typename AccessorType::value_type> buffer;
      buffer.resize(incomingAcc.getNChannels() * incomingAcc.getNElementsPerChannel());
      _continuousBuffer = buffer;
    }

    py::object get() const;

    std::string repr(py::object& acc) const;

    py::buffer_info getBufferInfo() const;

    py::object getattr(const std::string& name) const { return get().attr(name.c_str()); }
    py::object getitem(size_t index) const;

    static void bind(py::module& mod);

   private:
    mutable UserTypeTemplateVariantNoVoid<TwoDRegisterAccessor> _accessor;
    friend PyTransferElement;

    mutable UserTypeTemplateVariantNoVoid<Vector> _continuousBuffer;
    void copyToBuffer();
    void copyFromBuffer();
  };

  /********************************************************************************************************************/

} // namespace ChimeraTK
