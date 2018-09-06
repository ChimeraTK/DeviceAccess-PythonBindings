/*
 * new_class.cc
 *
 *  Created on: Sep 3, 2018
 */

#include "new_class.h"
#include <boost/variant.hpp>

using Element = boost::variant<IntegralType,      //
                               FloatingPointType, //
                               BooleanType,       //
                               StringType>;
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
}

/*****************************************************************************/
ElementStore getDefaultElementStore(Register::Type t, Register::Shape s) {
  switch (t) {
    case Register::Type::Bool:
      return ElementStore(
          s.rows, //
          std::vector<Element>(s.columns, BooleanType()));

    case Register::Type::FloatingPoint:
      return ElementStore(
          s.rows, //
          std::vector<Element>(s.columns, FloatingPointType()));

    case Register::Type::Integer:
      return ElementStore(s.rows, //
                          std::vector<Element>(s.columns, IntegralType()));

    case Register::Type::String:
      return ElementStore(
          s.rows, //
          std::vector<Element>(s.columns, Element{StringType()}));
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
// template specilizations
/*****************************************************************************/
template std::vector<std::vector<IntegralType> > //
pad(std::vector<std::vector<IntegralType> > v);
template std::vector<std::vector<FloatingPointType> > //
pad(std::vector<std::vector<FloatingPointType> > v);
template std::vector<std::vector<BooleanType> > //
pad(std::vector<std::vector<BooleanType> > v);
template std::vector<std::vector<StringType> > //
pad(std::vector<std::vector<StringType> > v);

template Register::Impl::Impl(std::string const& name, Access mode,
                              std::vector<std::vector<IntegralType> > data);
template Register::Impl::Impl(std::string const& name, Access mode,
                              std::vector<std::vector<FloatingPointType> > data);
template Register::Impl::Impl(std::string const& name, Access mode,
                              std::vector<std::vector<BooleanType> > data);
template Register::Impl::Impl(std::string const& name, Access mode,
                              std::vector<std::vector<StringType> > data);

template Register::Register(std::string const& name, Register::Access access,
                            std::vector<std::vector<IntegralType> > data);
template Register::Register(std::string const& name, Register::Access access,
                            std::vector<std::vector<FloatingPointType> > data);
template Register::Register(std::string const& name, Register::Access access,
                            std::vector<std::vector<BooleanType> > data);
template Register::Register(std::string const& name, Register::Access access,
                            std::vector<std::vector<StringType> > data);

template std::vector<std::vector<int> > Register::read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();
} // namespace TestBackend
