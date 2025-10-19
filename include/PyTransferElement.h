// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <pybind11/pybind11.h>
// pybind11.h must come first

// This needs to be included in every compilation unit which might define/use type conversions between Python and C++
// types, because it will define such converters. If the include is missing in some compilation unit, runtime crashes
// (stack smashing) might occur depending on the order of linkage.
#include "HelperFunctions.h" // IWYU pragma: keep

#include <ChimeraTK/AccessMode.h>
#include <ChimeraTK/TransferElementAbstractor.h>
#include <ChimeraTK/VersionNumber.h>

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

    static const std::set<std::string> specialAssignmentFunctionsToEmulateNumeric;

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

    void write(const ChimeraTK::VersionNumber& versionNumber = ChimeraTK::VersionNumber{}) {
      py::gil_scoped_release release;
      if(versionNumber == ChimeraTK::VersionNumber{nullptr}) {
        // If no version number is given, we use the default one
        visit([](auto& acc) { acc.write(); });
        return;
      }
      visit([&](auto& acc) { acc.write(versionNumber); });
    }
    void writeDestructively(const ChimeraTK::VersionNumber& versionNumber = ChimeraTK::VersionNumber{}) {
      py::gil_scoped_release release;
      if(versionNumber == ChimeraTK::VersionNumber{nullptr}) {
        // If no version number is given, we use the default one
        visit([](auto& acc) { acc.write(); });
        return;
      }
      visit([&](auto& acc) { acc.writeDestructively(versionNumber); });
    }

    void interrupt() { getTE().interrupt(); }

    auto getName() const { return getTE().getName(); }
    auto getUnit() const { return getTE().getUnit(); }

    [[nodiscard]] py::list getAccessModeFlags() const { return accessModeFlagsToList(getTE().getAccessModeFlags()); }

    auto getDescription() const { return getTE().getDescription(); }
    [[nodiscard]] py::dtype getValueType() const { return convertUsertypeToDtype(getTE().getValueType()); }
    [[nodiscard]] PyVersionNumber getVersionNumber() const { return getTE().getVersionNumber(); }
    auto isReadOnly() const { return getTE().isReadOnly(); }
    auto isReadable() const { return getTE().isReadable(); }
    auto isWriteable() const { return getTE().isWriteable(); }
    auto getId() const { return getTE().getId(); }
    auto dataValidity() const { return getTE().dataValidity(); }
    auto isInitialised() const { return getTE().isInitialised(); }
    auto setDataValidity(ChimeraTK::DataValidity validity) { return getTE().setDataValidity(validity); }

    // Note: using this function will bypass code added in ApplicationCore's ScalarAccessor/ArrayAccessor classes
    // and instead run functions as defined in DeviceAccess. Do not use for write operations.
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
    auto visit(CALLABLE fn) const {
      const auto* self = static_cast<const DerivedAccessor*>(this);
      return std::visit(fn, self->_accessor);
    }
  };

  /********************************************************************************************************************/

  template<template<typename> class AccessorType>
  class AccessorTypeTag {};

  /********************************************************************************************************************/

} // namespace ChimeraTK
