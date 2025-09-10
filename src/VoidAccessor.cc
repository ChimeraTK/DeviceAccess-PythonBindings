// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "VoidAccessor.h"

#include <ChimeraTK/cppext/finally.hpp>

#include <boost/python/numpy.hpp>

#include <ceval.h>
#include <pytypedefs.h>

namespace DeviceAccessPython {

  /********************************************************************************************************************/

  bool VoidRegisterAccessor::write(ChimeraTK::VoidRegisterAccessor& self) {
    return self.write();
  }

  /********************************************************************************************************************/

  bool VoidRegisterAccessor::writeDestructively(ChimeraTK::VoidRegisterAccessor& self) {
    return self.writeDestructively();
  }

  /********************************************************************************************************************/

  void VoidRegisterAccessor::read(ChimeraTK::VoidRegisterAccessor& self) {
    PyThreadState* m_thread_state = PyEval_SaveThread();
    auto _release = cppext::finally([&] { PyEval_RestoreThread(m_thread_state); });
    self.read();
  }

  /********************************************************************************************************************/

  bool VoidRegisterAccessor::readNonBlocking(ChimeraTK::VoidRegisterAccessor& self) {
    return self.readNonBlocking();
  }

  /********************************************************************************************************************/

  bool VoidRegisterAccessor::readLatest(ChimeraTK::VoidRegisterAccessor& self) {
    return self.readLatest();
  }

  /********************************************************************************************************************/

} // namespace DeviceAccessPython
