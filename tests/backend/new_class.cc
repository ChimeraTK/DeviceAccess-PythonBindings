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
ElementStore getElementStoreDefaults(Register::Type t, Register::Shape s);

template <typename UserType>
Register::Shape extractShape(std::vector<std::vector<UserType> > const& d);

/***************************************************************************/
template <typename T> Element castToVariantType(Register::Type t, T& value);

struct Register::Impl {

  /***************************************************************************/
  template <typename VariantType>
  Impl(std::string const& name, Access access,
       std::vector<std::vector<VariantType> > data)
      : name_(name), //
        access_(static_cast<Access>(access)) {

    if (!isValidShape(extractShape(data))) {
      std::logic_error("Invalid register shape requested; register must "
                       "accomodate at least one element");
    }
    for (auto& row : data) {
      elementStore_.emplace_back(std::vector<Element>{});
      for (auto element : row) {
        elementStore_.back().emplace_back(Element(element));
      }
    }
  }

  /***************************************************************************/
  Impl(std::string const& name, Access access, Type type, Shape shape)
      : name_(name), access_(static_cast<Access>(access)), elementStore_() {

    if (!isValidShape(shape)) {
      std::logic_error("Invalid register shape requested; register must "
                       "accomodate at least one element");
    }
    elementStore_ = getElementStoreDefaults(type, shape);
  }

  /***************************************************************************/
  bool isValidShape(Shape const& s) {
    return (s.rows != 0) && (s.columns != 0);
  }

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
bool Register::Shape::operator==(Shape const& rhs) {
  return (rows == rhs.rows) && (columns == rhs.columns);
}

/*****************************************************************************/
bool Register::Shape::operator!=(Shape const& rhs) { return !operator==(rhs); }

/*****************************************************************************/
template <typename UserType>
std::vector<std::vector<UserType> > Register::read() {

  auto& elementStore = impl_->elementStore_;
  std::vector<std::vector<UserType> > result;
  auto converter = [](auto& e) { return static_cast<UserType>(e); };

  for (auto& row : elementStore) {
    result.emplace_back(std::vector<UserType>());
    for (auto& element : row) {
      result.back().push_back(boost::apply_visitor(converter, element));
    }
  }
  return result;
}

/*****************************************************************************/
template <typename T> //
void Register::write(std::vector<std::vector<T> > data) {

  auto& elementStore = impl_->elementStore_;
  auto registerType = getType();
  auto registerShape = extractShape(elementStore);
  auto dataShape = extractShape(data);

  if (registerShape != dataShape) {
    throw std::runtime_error(
        "Input container does not confirm to Register shape");
  }
  for (size_t row = 0; row < registerShape.rows; row++) {
    for (size_t column = 0; column < registerShape.columns; column++) {
      elementStore[row][column] =
          castToVariantType(registerType, data[row][column]);
    }
  }
}

/*****************************************************************************/
Register::Type Register::getType() {
  return static_cast<Type>(impl_->elementStore_[0][0].which());
}

/*****************************************************************************/
Register::Shape Register::getShape() {
  return extractShape(impl_->elementStore_);
}

/*****************************************************************************/
Register::Access Register::getAccessMode() { return impl_->access_; }

/*****************************************************************************/
template <typename UserType>
Register::Shape extractShape(std::vector<std::vector<UserType> > const& d) {
  return Register::Shape{ d.size(), (d.size()) ? d[0].size() : 0 };
}

/*****************************************************************************/
ElementStore getElementStoreDefaults(Register::Type t, Register::Shape s) {
  switch (t) {
    case Register::Type::Bool:
      return ElementStore(s.rows, //
                          std::vector<Element>(s.columns, BooleanType()));

    case Register::Type::FloatingPoint:
      return ElementStore(s.rows, //
                          std::vector<Element>(s.columns, FloatingPointType()));

    case Register::Type::Integer:
      return ElementStore(s.rows, //
                          std::vector<Element>(s.columns, IntegralType()));

    case Register::Type::String:
      return ElementStore(
          s.rows, //
          std::vector<Element>(s.columns, Element{ StringType() }));
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

/***************************************************************************/
template <typename T> Element castToVariantType(Register::Type t, T& value) {
  switch (t) {
    case Register::Type::Integer:
      return static_cast<IntegralType>(value);
    case Register::Type::FloatingPoint:
      return static_cast<FloatingPointType>(value);
    case Register::Type::Bool:
      return static_cast<BooleanType>(value);
    case Register::Type::String:
      return static_cast<StringType>(value);
  }
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
template Register::Impl::Impl(
    std::string const& name, Access mode,
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
// template std::vector<std::vector<int> > read();
// template std::vector<std::vector<int> > read();
// template std::vector<std::vector<int> > read();
// template std::vector<std::vector<int> > read();
// template std::vector<std::vector<int> > read();
// template std::vector<std::vector<int> > read();
// template std::vector<std::vector<int> > read();
// template std::vector<std::vector<int> > read();
// template std::vector<std::vector<int> > read();
// template std::vector<std::vector<int> > read();
// template std::vector<std::vector<int> > read();

template void Register::write(std::vector<std::vector<int> > data);
} // namespace TestBackend
