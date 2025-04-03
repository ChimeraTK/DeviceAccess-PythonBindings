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

    static pybind11::array read(ACCESSOR& self, pybind11::array& np_buffer);

    static auto readNonBlocking(ACCESSOR& self, pybind11::array& np_buffer);

    static auto readLatest(ACCESSOR& self, pybind11::array& np_buffer);

    static bool write(ACCESSOR& self, pybind11::array& np_buffer);

    static bool writeDestructively(ACCESSOR& self, pybind11::array& np_buffer);

    template<typename UserType>
    static pybind11::array setAsCooked(
        ACCESSOR& self, pybind11::array& np_buffer, size_t channel, size_t element, UserType value);

    template<typename UserType>
    static UserType getAsCooked(ACCESSOR& self, pybind11::array& np_buffer, size_t channel, size_t element);
  };

  /*****************************************************************************************************************/
  /*****************************************************************************************************************/
  /* Implementations following */
  /*****************************************************************************************************************/
  /*****************************************************************************************************************/

  template<typename ACCESSOR>
  pybind11::array GeneralRegisterAccessor<ACCESSOR>::read(ACCESSOR& self, pybind11::array& np_buffer) {
    {
      PyThreadState* m_thread_state = PyEval_SaveThread();
      auto _release = cppext::finally([&] { PyEval_RestoreThread(m_thread_state); });
      self.read();
    }
    return copyUserBufferToNpArray(self, np_buffer);
  }

  /*****************************************************************************************************************/

  template<typename ACCESSOR>
  auto GeneralRegisterAccessor<ACCESSOR>::readNonBlocking(ACCESSOR& self, pybind11::array& np_buffer) {
    bool status = self.readNonBlocking();
    return pybind11::make_tuple(status, copyUserBufferToNpArray(self, np_buffer));
  }

  /*****************************************************************************************************************/

  template<typename ACCESSOR>
  auto GeneralRegisterAccessor<ACCESSOR>::readLatest(ACCESSOR& self, pybind11::array& np_buffer) {
    bool status = self.readLatest();
    return pybind11::make_tuple(status, copyUserBufferToNpArray(self, np_buffer));
  }

  /*****************************************************************************************************************/

  template<typename ACCESSOR>
  bool GeneralRegisterAccessor<ACCESSOR>::write(ACCESSOR& self, pybind11::array& np_buffer) {
    copyNpArrayToUserBuffer(self, np_buffer);
    return self.write();
  }

  /*****************************************************************************************************************/

  template<typename ACCESSOR>
  bool GeneralRegisterAccessor<ACCESSOR>::writeDestructively(ACCESSOR& self, pybind11::array& np_buffer) {
    copyNpArrayToUserBuffer(self, np_buffer);
    return self.writeDestructively();
  }

  /*****************************************************************************************************************/

  template<typename ACCESSOR>
  template<typename UserType>
  pybind11::array GeneralRegisterAccessor<ACCESSOR>::setAsCooked(
      ACCESSOR& self, pybind11::array& np_buffer, size_t channel, size_t element, UserType value) {
    self.getImpl()->setAsCooked(channel, element, value);
    return copyUserBufferToNpArray(self, np_buffer);
  }
  /*****************************************************************************************************************/

  template<typename ACCESSOR>
  template<typename UserType>
  UserType GeneralRegisterAccessor<ACCESSOR>::getAsCooked(
      ACCESSOR& self, pybind11::array& np_buffer, size_t channel, size_t element) {
    copyNpArrayToUserBuffer(self, np_buffer);
    return self.getImpl()->template getAsCooked<UserType>(channel, element);
  }

  /*****************************************************************************************************************/

} // namespace DeviceAccessPython
