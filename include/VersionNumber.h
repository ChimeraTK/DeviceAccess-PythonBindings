// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <ChimeraTK/VersionNumber.h>

#include <boost/date_time/posix_time/posix_time_types.hpp>

namespace DeviceAccessPython {

  class VersionNumber {
   public:
    static boost::posix_time::ptime getTime(ChimeraTK::VersionNumber& self);
    static ChimeraTK::VersionNumber getNullVersion();
  };

} // namespace DeviceAccessPython