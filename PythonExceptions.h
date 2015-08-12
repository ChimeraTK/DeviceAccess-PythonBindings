#ifndef SOURCE_DIRECTORY__PYTHONEXCEPTION_H_
#define SOURCE_DIRECTORY__PYTHONEXCEPTION_H_

#include <exception>

#define MTCA4U_PYTHON_EXCEPTION(NAME)                                          \
  class NAME : public Exception {                                              \
  public:                                                                      \
    NAME(std::string const &message) : Exception(message) {}                   \
  };

namespace mtca4upy {

class Exception : public std::exception {
  std::string _errorMsg;

public:
  Exception(std::string const &message) : _errorMsg(message) {}
  virtual const char *what() const throw() { return _errorMsg.c_str(); }
  virtual ~Exception() throw() {}
};
MTCA4U_PYTHON_EXCEPTION(ArrayOutOfBoundException)
MTCA4U_PYTHON_EXCEPTION(MethodNotImplementedException)
MTCA4U_PYTHON_EXCEPTION(DeviceNotSupported)
MTCA4U_PYTHON_EXCEPTION(DummyDeviceBadParameterException)
MTCA4U_PYTHON_EXCEPTION(ArrayElementTypeNotSupported)
}

#endif /* SOURCE_DIRECTORY__PYTHONEXCEPTION_H_ */
