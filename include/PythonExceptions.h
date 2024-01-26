// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <exception>
#include <string>

#define MTCA4U_PYTHON_EXCEPTION(NAME)                                                                                  \
  class NAME : public Exception {                                                                                      \
   public:                                                                                                             \
    NAME(std::string const& message) : Exception(message) {}                                                           \
  };

namespace DeviceAccessPython {

  class Exception : public std::exception {
    std::string _errorMsg;

   public:
    Exception(std::string const& message) : _errorMsg(message) {}
    virtual const char* what() const throw() { return _errorMsg.c_str(); }
    virtual ~Exception() throw() {}
  };

  MTCA4U_PYTHON_EXCEPTION(ArrayOutOfBoundException)
  MTCA4U_PYTHON_EXCEPTION(ArrayElementTypeNotSupported)

} // namespace DeviceAccessPython
