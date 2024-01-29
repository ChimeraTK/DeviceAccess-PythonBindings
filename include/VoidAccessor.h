// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <ChimeraTK/Device.h>

namespace DeviceAccessPython {

  /**
   * (Static) class implementing special functions for Void register accessors
   */
  class VoidRegisterAccessor {
   public:
    static bool write(ChimeraTK::VoidRegisterAccessor& self);

    static void read(ChimeraTK::VoidRegisterAccessor& self);

    static bool readNonBlocking(ChimeraTK::VoidRegisterAccessor& self);

    static bool readLatest(ChimeraTK::VoidRegisterAccessor& self);

    static bool writeDestructively(ChimeraTK::VoidRegisterAccessor& self);
  };

} // namespace DeviceAccessPython
