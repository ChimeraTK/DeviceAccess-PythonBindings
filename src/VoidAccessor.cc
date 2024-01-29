// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "VoidAccessor.h"

namespace mtca4upy {

  /*****************************************************************************************************************/

  bool VoidRegisterAccessor::write(ChimeraTK::VoidRegisterAccessor& self) {
    return self.write();
  }

  /*****************************************************************************************************************/

  bool VoidRegisterAccessor::writeDestructively(ChimeraTK::VoidRegisterAccessor& self) {
    return self.writeDestructively();
  }

  /*****************************************************************************************************************/

  void VoidRegisterAccessor::read(ChimeraTK::VoidRegisterAccessor& self) {
    return self.read();
  }

  /*****************************************************************************************************************/

  bool VoidRegisterAccessor::readNonBlocking(ChimeraTK::VoidRegisterAccessor& self) {
    return self.readNonBlocking();
  }

  /*****************************************************************************************************************/

  bool VoidRegisterAccessor::readLatest(ChimeraTK::VoidRegisterAccessor& self) {
    return self.readLatest();
  }

  /*****************************************************************************************************************/

} // namespace mtca4upy