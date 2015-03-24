#ifndef HELPERFUNCTIONS_H_
#define HELPERFUNCTIONS_H_
#include <boost/python.hpp>

namespace bp = boost::python;

namespace mtca4upy{

  float* extractDataPointer(const bp::numeric::array& Buffer);

}


#endif /* HELPERFUNCTIONS_H_ */
