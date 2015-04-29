#ifndef SOURCE_DIRECTORY__PYTHONEXCEPTION_H_
#define SOURCE_DIRECTORY__PYTHONEXCEPTION_H_

#include <exception>

namespace mtca4upy { // TODO: Refactor to a better name

class ArrayOutOfBoundException : public std::exception {
public:
  inline virtual const char *what() const throw() {
    return "size to write is more than the supplied array size";
  }
};

class MethodNotImplementedException : public std::exception {
public:
  inline virtual const char *what() const throw() {
    return "This method is not available for this device";
  }
};


class DeviceNotSupported : public std::exception {
public:
  inline virtual const char *what() const throw() {
    return "Unable to identify device";
  }
};

class DummyDeviceBadParameterException : public std::exception {
public:
  inline virtual const char* what() const throw() {
    return "Mapped Dummy Device expects first and second parameters to be the "
           "same map file";
  }
};

class ArrayElementTypeNotSupported : public std::exception {
public:
  inline virtual const char *what() const throw() {
    return "Numpy array dtype used is not supported for this method";
  }
};


}

#endif /* SOURCE_DIRECTORY__PYTHONEXCEPTION_H_ */
