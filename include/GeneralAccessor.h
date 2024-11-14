// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "HelperFunctions.h"

#include <ChimeraTK/cppext/finally.hpp>

namespace DeviceAccessPython {

  /*****************************************************************************************************************/

  /**
   * (Static) class implementing register accessor functions for all accessor types
   */
  template<typename ACCESSOR>
  class GeneralRegisterAccessor {
   public:
    // "convert" the return type from a const string reference to a real string, since boost python cannot deal with
    // that otherwise
    static std::string getName(ACCESSOR& self) { return self.getName(); }

    static std::string getUnit(ACCESSOR& self) { return self.getUnit(); }

    static std::string getDescription(ACCESSOR& self) { return self.getDescription(); }

    static std::string getAccessModeFlagsString(ACCESSOR& self) { return self.getAccessModeFlags().serialize(); }

    static boost::python::numpy::ndarray read(ACCESSOR& self, boost::python::numpy::ndarray& np_buffer);

    static auto readNonBlocking(ACCESSOR& self, boost::python::numpy::ndarray& np_buffer);

    static auto readLatest(ACCESSOR& self, boost::python::numpy::ndarray& np_buffer);

    static bool write(ACCESSOR& self, boost::python::numpy::ndarray& np_buffer);

    static bool writeDestructively(ACCESSOR& self, boost::python::numpy::ndarray& np_buffer);

    template<typename UserType>
    static boost::python::numpy::ndarray setAsCooked(
        ACCESSOR& self, boost::python::numpy::ndarray& np_buffer, size_t channel, size_t element, UserType value);

    template<typename UserType>
    static UserType getAsCooked(
        ACCESSOR& self, boost::python::numpy::ndarray& np_buffer, size_t channel, size_t element);
  };

  /*****************************************************************************************************************/
  /*****************************************************************************************************************/
  /* Implementations following */
  /*****************************************************************************************************************/
  /*****************************************************************************************************************/

  template<typename ACCESSOR>
  boost::python::numpy::ndarray GeneralRegisterAccessor<ACCESSOR>::read(
      ACCESSOR& self, boost::python::numpy::ndarray& np_buffer) {
    {
      PyThreadState* m_thread_state = PyEval_SaveThread();
      auto _release = cppext::finally([&] { PyEval_RestoreThread(m_thread_state); });
      self.read();
    }
    return copyUserBufferToNpArray(self, np_buffer);
  }

  /*****************************************************************************************************************/

  template<typename ACCESSOR>
  auto GeneralRegisterAccessor<ACCESSOR>::readNonBlocking(ACCESSOR& self, boost::python::numpy::ndarray& np_buffer) {
    bool status = self.readNonBlocking();
    return boost::python::make_tuple(status, copyUserBufferToNpArray(self, np_buffer));
  }

  /*****************************************************************************************************************/

  template<typename ACCESSOR>
  auto GeneralRegisterAccessor<ACCESSOR>::readLatest(ACCESSOR& self, boost::python::numpy::ndarray& np_buffer) {
    bool status = self.readLatest();
    return boost::python::make_tuple(status, copyUserBufferToNpArray(self, np_buffer));
  }

  /*****************************************************************************************************************/

  template<typename ACCESSOR>
  bool GeneralRegisterAccessor<ACCESSOR>::write(ACCESSOR& self, boost::python::numpy::ndarray& np_buffer) {
    copyNpArrayToUserBuffer(self, np_buffer);
    return self.write();
  }

  /*****************************************************************************************************************/

  template<typename ACCESSOR>
  bool GeneralRegisterAccessor<ACCESSOR>::writeDestructively(ACCESSOR& self, boost::python::numpy::ndarray& np_buffer) {
    copyNpArrayToUserBuffer(self, np_buffer);
    return self.writeDestructively();
  }

  /*****************************************************************************************************************/

  template<typename ACCESSOR>
  template<typename UserType>
  boost::python::numpy::ndarray GeneralRegisterAccessor<ACCESSOR>::setAsCooked(
      ACCESSOR& self, boost::python::numpy::ndarray& np_buffer, size_t channel, size_t element, UserType value) {
    self.getImpl()->setAsCooked(channel, element, value);
    return copyUserBufferToNpArray(self, np_buffer);
  }
  /*****************************************************************************************************************/

  template<typename ACCESSOR>
  template<typename UserType>
  UserType GeneralRegisterAccessor<ACCESSOR>::getAsCooked(
      ACCESSOR& self, boost::python::numpy::ndarray& np_buffer, size_t channel, size_t element) {
    copyNpArrayToUserBuffer(self, np_buffer);
    return self.getImpl()->template getAsCooked<UserType>(channel, element);
  }

  /*****************************************************************************************************************/

} // namespace DeviceAccessPython
