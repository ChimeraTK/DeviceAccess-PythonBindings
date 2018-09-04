/*
 * new_class.cc
 *
 *  Created on: Sep 3, 2018
 */

#include "new_class.h"
#include <boost/variant.hpp>

using Variant = boost::variant<int, double, bool, std::string>;
using Data = std::vector<std::vector<Variant> >;

namespace TestBackend {

static Data fillDefaults(Register::Type t, Register::Shape s) {
  // return Data{ s.rows, { s.columns, {static_cast<bool>(false)} } };
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
  Data data_;

  template <typename VariantType>
  Impl(std::string const& name, Access access,
       std::vector<std::vector<VariantType> > data)
      : name_(name),                         //
        access_(static_cast<Access>(access)) //
  {
    for (auto& row : data) {
      data_.emplace_back(std::vector<Variant>{});
      for (auto element : row) {
        data_.back().emplace_back(Variant(element));
      }
    }
  }
};

template <typename VariantType>
Register::Register(std::string const& name, //
                   Register::Access access, //
                   std::vector<std::vector<VariantType> > data)
    : impl_(std::make_unique<Impl>(name, access, data)) {}

// To get around the static assertion with unique ptr
Register::~Register() = default;

// template specilization
template Register::Impl::Impl(std::string const& name, Access mode,
                              std::vector<std::vector<bool> > data);
template Register::Impl::Impl(std::string const& name, Access mode,
                              std::vector<std::vector<int> > data);
template Register::Impl::Impl(std::string const& name, Access mode,
                              std::vector<std::vector<double> > data);
template Register::Impl::Impl(std::string const& name, Access mode,
                              std::vector<std::vector<std::string> > data);

template Register::Register(std::string const& name, Register::Access access,
                            std::vector<std::vector<bool> > data);
template Register::Register(std::string const& name, Register::Access access,
                            std::vector<std::vector<int> > data);
template Register::Register(std::string const& name, Register::Access access,
                            std::vector<std::vector<double> > data);
template Register::Register(std::string const& name, Register::Access access,
                            std::vector<std::vector<std::string> > data);
} // namespace TestBackend
