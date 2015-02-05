/*
 * PythonException.h
 *
 *  Created on: Feb 3, 2015
 *      Author: varghese
 */

#ifndef SOURCE_DIRECTORY__PYTHONEXCEPTION_H_
#define SOURCE_DIRECTORY__PYTHONEXCEPTION_H_

#include <exception>

namespace mtca4upy { // TODO: Refactor to a better name

class ArrayOutOfBoundException : public std::exception {
public:
  inline virtual const char *what() const throw() {
    return "size to write is more than the specified array size";
  }
};

class MethodNotImplementedException : public std::exception {
public:
  inline virtual const char *what() const throw() {
    return "This method is not available for this device";
  }
};

class DummyDeviceBadParameterException : public std::exception {
public:
  inline virtual const char *what() const throw() {
    return "Mapped Dummy Device expects first and second parameters to be the "
           "same map file";
  }
};
}

#endif /* SOURCE_DIRECTORY__PYTHONEXCEPTION_H_ */
