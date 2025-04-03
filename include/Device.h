// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <ChimeraTK/Device.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <boost/python/numpy.hpp>
namespace py = pybind11;

namespace DeviceAccessPython {

  /** (Static) class to map ChimeraTK::Device to python */
  class Device {
   public:
    static ChimeraTK::TwoDRegisterAccessor<double> getTwoDAccessor(
        const ChimeraTK::Device& self, const std::string& registerPath);

    static ChimeraTK::AccessModeFlags convertFlagsFromPython(const py::list& flaglist);

    template<typename T>
    static ChimeraTK::TwoDRegisterAccessor<T> getGeneralTwoDAccessor(const ChimeraTK::Device& self,
        const std::string& registerPath, size_t numberOfElements, size_t elementsOffset, const py::list& flaglist) {
      return self.getTwoDRegisterAccessor<T>(
          registerPath, numberOfElements, elementsOffset, convertFlagsFromPython(flaglist));
    }

    template<typename T>
    static ChimeraTK::OneDRegisterAccessor<T> getGeneralOneDAccessor(const ChimeraTK::Device& self,
        const std::string& registerPath, size_t numberOfElements, size_t elementsOffset, const py::list& flaglist) {
      return self.getOneDRegisterAccessor<T>(
          registerPath, numberOfElements, elementsOffset, convertFlagsFromPython(flaglist));
    }

    template<typename T>
    static ChimeraTK::ScalarRegisterAccessor<T> getGeneralScalarAccessor(const ChimeraTK::Device& self,
        const std::string& registerPath, size_t elementsOffset, const py::list& flaglist) {
      return self.getScalarRegisterAccessor<T>(registerPath, elementsOffset, convertFlagsFromPython(flaglist));
    }

    static ChimeraTK::VoidRegisterAccessor getVoidRegisterAccessor(
        const ChimeraTK::Device& self, const std::string& registerPath, const py::list& flaglist);

    template<typename T>
    static ChimeraTK::OneDRegisterAccessor<T> getOneDAccessor(const ChimeraTK::Device& self,
        const std::string& registerPath, size_t numberOfelementsToRead, size_t elementOffset) {
      return self.getOneDRegisterAccessor<T>(registerPath, numberOfelementsToRead, elementOffset);
    }

    static ChimeraTK::OneDRegisterAccessor<int32_t> getRawOneDAccessor(const ChimeraTK::Device& self,
        const std::string& registerPath, size_t numberOfelementsToRead, size_t elementOffset);

    static std::string getCatalogueMetadata(const ChimeraTK::Device& self, const std::string& parameterName);

    static void open(ChimeraTK::Device& self, std::string const& aliasName);
    static void open(ChimeraTK::Device& self);
    static void close(ChimeraTK::Device& self);

    static void activateAsyncRead(ChimeraTK::Device& self);
    static ChimeraTK::RegisterCatalogue getRegisterCatalogue(ChimeraTK::Device& self);

    static void write(const ChimeraTK::Device& self, py::array& arr, const std::string& registerPath,
        size_t numberOfElements, size_t elementsOffset, const py::list& flaglist);

    static boost::python::numpy::ndarray read(const ChimeraTK::Device& self, const std::string& registerPath,
        size_t numberOfElements, size_t elementsOffset, const py::list& flaglist);
  };

} // namespace DeviceAccessPython
