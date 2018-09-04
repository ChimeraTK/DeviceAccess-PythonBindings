/*
 * new_class.cc
 *
 *  Created on: Sep 3, 2018
 */

#include "new_class.h"
#include <boost/variant.hpp>

using Data =
    std::vector<std::vector<boost::variant<int, double, bool, std::string> > >;

namespace TestBackend {

template <typename UserType>
static Register::Shape getShape(std::vector<std::vector<UserType> > const& data) {};

static Data fillDefaults(Register::Type t, Register::Shape s) {
   //return Data{ s.rows, { s.columns, {static_cast<bool>(false)} } };
/*  switch (t) {
    case RegisterType::Bool:
      //return Data{ s.rows, { s.columns, static_cast<bool>(false) } };
    case RegisterType::Double:
      return Data{ s.rows, { s.columns, 0.0 } };
    case RegisterType::Int:
      return Data{ s.rows, { s.columns, static_cast<int>(0) } };
    case RegisterType::String:
      return Data{ s.rows, { s.columns, static_cast<std::string>("") } };
  }*/
}
struct Register::Impl {

  std::string name_;
  Access access_;
  Type registerType_;
  Data data_;

  Impl(std::string name, Access mode, Type type, Shape shape)
      : name_(name), access_(mode), registerType_(type) {
  }
};

// To get around the static assertion with unique ptr
Register::~Register() = default;

template <typename UserType>
Register::Register(std::string const& name, //
           Register::Access mode,         //
           std::vector<std::vector<UserType> > data){}

/*Register::Register(std::string const& name, //
           Register::Access mode,         //
           Register::Type type,       //
           Shape shape)
    : impl_(std::make_unique<Impl>(name, mode, type, shape)) {}*/

//template Register::Register<int>(std::string const& name, //
                         //Register::Access mode,         //
                         //std::vector<std::vector<int> > data);

//template static Register::Shape getShape(
    //std::vector<std::vector<int> > const& data);
//}
}
