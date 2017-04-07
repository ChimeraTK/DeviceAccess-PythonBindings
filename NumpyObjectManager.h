#ifndef NUMPYOBJECTMANAGER_H_
#define NUMPYOBJECTMANAGER_H_

#include <boost/python.hpp>
#include <numpy/arrayobject.h>

namespace bp = boost::python;

namespace mtca4upy {

// Had to drop boost::python::numeric to wrap numpy ndarray; using bp:numeric
// causes segfaults on python3 exit:
// https://github.com/boostorg/python/issues/75
// https://github.com/boostorg/python/issues/79
//
// Work around the problem by defining a custom python object to handle the
// numpy array. These links were used as reference to reach this solution:
// https://github.com/mantidproject/mantid/commit/7371d129eca58daa6ca1e8852c8de736f2cefc66
// https://github.com/boostorg/python/blob/352792c90a36ee8d6a520c7602b2dbd03ab19c3f/include/boost/python/numpy/numpy_object_mgr_traits.hpp

class NumpyObject : public boost::python::object {
public:
  BOOST_PYTHON_FORWARD_OBJECT_CONSTRUCTORS(NumpyObject, bp::object);
};
} // namespace: mtca4upy

namespace boost {
namespace python {
  namespace converter {

    template <> struct object_manager_traits<mtca4upy::NumpyObject> {
      BOOST_STATIC_CONSTANT(bool, is_specialized = true);

      /*
       * Return a boost::python::object handler when obj is a PyArray_Type
       */
      static inline python::detail::new_reference adopt(PyObject* obj) {
        return python::detail::new_reference(
            (python::pytype_check((PyTypeObject*)get_pytype(), obj)));
      }

      static inline bool check(PyObject* obj) {
        return PyObject_IsInstance(obj, (PyObject*)get_pytype());
      }

      static inline PyTypeObject const* get_pytype() { return &PyArray_Type; }
    };
  }
}
} // namespace: boost::python::converter

#endif /* NUMPYOBJECTMANAGER_H_ */
