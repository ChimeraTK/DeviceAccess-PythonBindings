// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "VersionNumber.h"

namespace mtca4upy {

  /*****************************************************************************************************************/

  boost::posix_time::ptime VersionNumber::getTime([[maybe_unused]] ChimeraTK::VersionNumber& self) {
    return boost::posix_time::ptime(boost::gregorian::date(1990, 1, 1));
  }

  /*****************************************************************************************************************/

  ChimeraTK::VersionNumber VersionNumber::getNullVersion() {
    return ChimeraTK::VersionNumber(nullptr);
  }

} // namespace mtca4upy