// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <pybind11/pybind11.h>
// pybind11.h must come first

#include <ChimeraTK/AccessMode.h>
#include <ChimeraTK/TransferElementAbstractor.h>

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/

  class PyTransferElementBase {
   public:
    virtual ~PyTransferElementBase() = default;

    virtual const TransferElementAbstractor& getTE() const = 0;
    virtual TransferElementAbstractor& getTE() = 0;

    static void bind(py::module& mod);

    static const std::set<std::string> specialFunctionsToEmulateNumeric;

    static const std::set<std::string> specialUnaryFunctionsToEmulateNumeric;
  };

  /********************************************************************************************************************/

  template<class DerivedAccessor>
  class PyTransferElement : public PyTransferElementBase {
   public:
    void read() {
      py::gil_scoped_release release;
      visit([](auto& acc) { acc.read(); });
    }
    bool readNonBlocking() {
      py::gil_scoped_release release;
      bool rv;
      visit([&](auto& acc) { rv = acc.readNonBlocking(); });
      return rv;
    }
    bool readLatest() {
      py::gil_scoped_release release;
      bool rv;
      visit([&](auto& acc) { rv = acc.readLatest(); });
      return rv;
    }
    void write() {
      py::gil_scoped_release release;
      visit([](auto& acc) { acc.write(); });
    }
    void writeDestructively() {
      py::gil_scoped_release release;
      visit([](auto& acc) { acc.writeDestructively(); });
    }

    auto getName() const { return getTE().getName(); }
    auto getUnit() const { return getTE().getUnit(); }
    py::list getAccessModeFlags() const {
      ChimeraTK::AccessModeFlags flags = getTE().getAccessModeFlags();
      py::list python_flags{};
      if(flags.has(ChimeraTK::AccessMode::raw)) python_flags.append(ChimeraTK::AccessMode::raw);
      if(flags.has(ChimeraTK::AccessMode::wait_for_new_data))
        python_flags.append(ChimeraTK::AccessMode::wait_for_new_data);
      return python_flags;
    }
    auto getDescription() const { return getTE().getDescription(); }
    DataType getValueType() const { return getTE().getValueType(); }
    auto getVersionNumber() const { return getTE().getVersionNumber(); }
    auto isReadOnly() const { return getTE().isReadOnly(); }
    auto isReadable() const { return getTE().isReadable(); }
    auto isWriteable() const { return getTE().isWriteable(); }
    auto getId() const { return getTE().getId(); }
    auto dataValidity() const { return getTE().dataValidity(); }

    // Note: using this function will bypass code added in ApplicationCore's ScalarAccessor/ArrayAccessor classes and
    // instead run functions as defined in DeviceAccess. Do not use for write operations.
    [[nodiscard]] const TransferElementAbstractor& getTE() const final {
      const auto* self = static_cast<const DerivedAccessor*>(this);
      TransferElementAbstractor* te;
      std::visit([&](auto& acc) { te = &acc; }, self->_accessor);
      return *te;
    }
    // non-const version which can be used to modify the original TransferElementAbstractor, e.g. for decoration
    [[nodiscard]] TransferElementAbstractor& getTE() final {
      const auto* self = static_cast<const DerivedAccessor*>(this);
      TransferElementAbstractor* te;
      std::visit([&](auto& acc) { te = &acc; }, self->_accessor);
      return *te;
    }

    template<typename AccessorType>
    void setTE(AccessorType incomingAcc) {
      const auto* self = static_cast<const DerivedAccessor*>(this);
      self->_accessor.template emplace<AccessorType>(incomingAcc);
    }

    // Pass the actual accessor type (e.g. ScalarAccessor<int>) as argument to the given callable
    template<typename CALLABLE>
    void visit(CALLABLE fn) const {
      const auto* self = static_cast<const DerivedAccessor*>(this);
      std::visit(fn, self->_accessor);
    }
  };

  /********************************************************************************************************************/

  template<template<typename> class AccessorType>
  class AccessorTypeTag {};

  /********************************************************************************************************************/

} // namespace ChimeraTK