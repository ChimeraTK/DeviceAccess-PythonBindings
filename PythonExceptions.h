#ifndef SOURCE_DIRECTORY__PYTHONEXCEPTION_H_
#define SOURCE_DIRECTORY__PYTHONEXCEPTION_H_

#include <exception>
#include <string>

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
MTCA4U_PYTHON_EXCEPTION(ArrayElementTypeNotSupported)
} // namespace mtca4upy

#endif /* SOURCE_DIRECTORY__PYTHONEXCEPTION_H_ */
