/*
 * new_class.cc
 *
 *  Created on: Sep 3, 2018
 */

#include "new_class.h"
#include "VariantTypes.h"
#include <boost/variant.hpp>

IntegralType a;

using Element = boost::variant<int, double, bool, std::string>;
using ElementStore = std::vector<std::vector<Element> >;

namespace TestBackend {

template <typename T>
std::vector<std::vector<T> > pad(std::vector<std::vector<T> > input);
ElementStore getDefaultElementStore(Register::Type t, Register::Shape s);

struct Register::Impl {

  /***************************************************************************/
  template <typename VariantType>
  Impl(std::string const& name, Access access,
       std::vector<std::vector<VariantType> > data)
      : name_(name), //
        access_(static_cast<Access>(access)) {
    for (auto& row : data) {
      elementStore_.emplace_back(std::vector<Element>{});
      for (auto element : row) {
        elementStore_.back().emplace_back(Element(element));
      }
    }
  }

  /***************************************************************************/
  Impl(std::string const& name, Access access, Type type, Shape shape)
      : name_(name),
        access_(static_cast<Access>(access)),
        elementStore_(getDefaultElementStore(type, shape)) {}

  /***************************************************************************/
  std::string name_;
  Access access_;
  ElementStore elementStore_;
};

/*****************************************************************************/
Register::Register(std::string const& name, //
                   Access access,           //
                   Type type,               //
                   Shape shape)
    : impl_(std::make_unique<Impl>(name, access, type, shape)) {}

/*****************************************************************************/
template <typename VariantType>
Register::Register(std::string const& name, //
                   Register::Access access, //
                   std::vector<std::vector<VariantType> > data)
    : impl_(std::make_unique<Impl>(name, access, pad(data))) {}

/*****************************************************************************/
// To get around the static assertion with unique ptr
Register::~Register() = default;

/*****************************************************************************/
template <typename UserType>
std::vector<std::vector<UserType>> Register::read(){
/*
  auto &elementStore = impl_->elementStore_;
  auto convert = [](auto& e){return static_cast<UserType>(e);};
  std::vector<std::vector<UserType>> result;

  for (auto & row: elementStore){
    result.emplace_back(std::vector<UserType>());
    for(auto & element: row){
      result.back().push_back(boost::apply_visitor(convert, element));  
    }
  }
  return result;
*/
}

/*****************************************************************************/
ElementStore getDefaultElementStore(Register::Type t, Register::Shape s) {
  switch (t) {
    case Register::Type::Bool:
      return ElementStore(
          s.rows, //
          std::vector<Element>(s.columns, static_cast<bool>(false)));

    case Register::Type::Double:
      return ElementStore(
          s.rows, //
          std::vector<Element>(s.columns, static_cast<double>(0.0)));

    case Register::Type::Int:
      return ElementStore(s.rows, //
                          std::vector<Element>(s.columns, static_cast<int>(0)));

    case Register::Type::String:
      return ElementStore(
          s.rows, //
          std::vector<Element>(s.columns, static_cast<std::string>("")));
  }
  return ElementStore();
}

/*****************************************************************************/
template <typename T>
std::vector<std::vector<T> > pad(std::vector<std::vector<T> > input) {
  std::size_t maxRowSize = 0;
  for (auto const& row : input) {
    if (row.size() > maxRowSize) {
      maxRowSize = row.size();
    }
  }
  for (auto& row : input) {
    row.resize(maxRowSize);
  }
  return input;
}

/*****************************************************************************/
// template specilization
template std::vector<std::vector<int> > //
pad(std::vector<std::vector<int> > v);
template std::vector<std::vector<bool> > //
pad(std::vector<std::vector<bool> > v);
template std::vector<std::vector<double> > //
pad(std::vector<std::vector<double> > v);
template std::vector<std::vector<std::string> > //
pad(std::vector<std::vector<std::string> > v);

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
