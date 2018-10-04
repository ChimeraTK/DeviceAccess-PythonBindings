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
using ElementStore = std::vector<std::vector<Element>>;

namespace TestBackend {

template <typename T>
std::vector<std::vector<T>> pad(std::vector<std::vector<T>> input);
ElementStore getElementStoreDefaults(Register::Type t, Register::Shape s);

template <typename UserType>
Register::Shape extractShape(std::vector<std::vector<UserType>> const &d);

/***************************************************************************/
template <typename T> Element castToVariantType(Register::Type t, T &&value);

struct Register::Impl {
  std::string name_;
  Access access_;
  ElementStore elementStore_;

  template <typename VariantType>
  Impl(std::string const &name, Access access,
       std::vector<std::vector<VariantType>> data)
      : name_(name), //
        access_(static_cast<Access>(access)) {

    // this call is to validate shape of data. extractShape throws
    // if data has an unsupported shape. Fix this kludge?
    extractShape(data);

    for (auto &row : data) {
      elementStore_.emplace_back(std::vector<Element>{});
      for (auto element : row) {
        elementStore_.back().emplace_back(Element(element));
      }
    }
  }

  Impl(std::string const &name, Access access, Type type, Shape shape)
      : name_(name), access_(static_cast<Access>(access)), elementStore_() {
    elementStore_ = getElementStoreDefaults(type, shape);
  }
};

Register::Shape::Shape(size_t r, size_t c) : rows_(r), columns_(c) {
  if (rows_ == 0 || columns_ == 0) {
    throw std::logic_error("Invalid register shape requested; register must "
                           "accomodate at least one element");
  }
}

/*****************************************************************************/
Register::Register(std::string const &name, //
                   Access access,           //
                   Type type,               //
                   Shape shape)
    : impl_(std::make_unique<Impl>(name, access, type, shape)) {}

/*****************************************************************************/
template <typename VariantType>
Register::Register(std::string const &name, //
                   Register::Access access, //
                   std::vector<std::vector<VariantType>> data)
    : impl_(std::make_unique<Impl>(name, access, pad(data))) {}

/*****************************************************************************/
Register::Register(Register &&r) : impl_(std::move(r.impl_)) {}
/*****************************************************************************/
// To get around the static assertion with unique ptr
Register::~Register() = default;
Register::View::~View() = default;

/*****************************************************************************/
bool Register::Shape::operator==(Shape const &rhs) {
  return (rows_ == rhs.rows_) && (columns_ == rhs.columns_);
}

/*****************************************************************************/
bool Register::Shape::operator!=(Shape const &rhs) { return !operator==(rhs); }

/*****************************************************************************/
template <typename UserType>
std::vector<std::vector<UserType>> Register::read() {
  auto &elementStore = impl_->elementStore_;
  std::vector<std::vector<UserType>> result;
  auto converter = [](auto &e) { return static_cast<UserType>(e); };

  for (auto &row : elementStore) {
    result.emplace_back(std::vector<UserType>());
    for (auto &element : row) {
      result.back().push_back(boost::apply_visitor(converter, element));
    }
  }
  return result;
}

/*****************************************************************************/
template <typename T> //
void Register::write(std::vector<std::vector<T>> const &data) {
  auto &elementStore = impl_->elementStore_;
  auto registerType = getType();
  auto registerShape = extractShape(elementStore);
  auto dataShape = extractShape(data);

  if (registerShape != dataShape) {
    throw std::runtime_error(
        "Input container does not confirm to Register shape");
  }
  auto rowSize = registerShape.getRows();
  auto columnSize = registerShape.getColumns();

  for (size_t row = 0; row < rowSize; row++) {
    for (size_t column = 0; column < columnSize; column++) {
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
std::string Register::getName() { return impl_->name_; }

Register::View Register::getView(Window w) {}
/*****************************************************************************/
Register::Access Register::getAccessMode() { return impl_->access_; }

/*****************************************************************************/
template <typename UserType>
Register::Shape extractShape(std::vector<std::vector<UserType>> const &d) {
  return Register::Shape{d.size(), (d.size()) ? d[0].size() : 0};
}

/*****************************************************************************/
ElementStore getElementStoreDefaults(Register::Type t, Register::Shape s) {
  switch (t) {
  case Register::Type::Bool:
    return ElementStore(s.getRows(), //
                        std::vector<Element>(s.getColumns(), BooleanType()));

  case Register::Type::FloatingPoint:
    return ElementStore(
        s.getRows(), //
        std::vector<Element>(s.getColumns(), FloatingPointType()));

  case Register::Type::Integer:
    return ElementStore(s.getRows(), //
                        std::vector<Element>(s.getColumns(), IntegralType()));

  case Register::Type::String:
    return ElementStore(
        s.getRows(), //
        std::vector<Element>(s.getColumns(), Element{StringType()}));
  }
  return ElementStore();
}

/*****************************************************************************/
template <typename T>
std::vector<std::vector<T>> pad(std::vector<std::vector<T>> input) {
  std::size_t maxRowSize = 0;
  for (auto const &row : input) {
    if (row.size() > maxRowSize) {
      maxRowSize = row.size();
    }
  }
  for (auto &row : input) {
    row.resize(maxRowSize);
  }
  return input;
}

/***************************************************************************/
template <typename T> Element castToVariantType(Register::Type t, T &&value) {
  switch (t) {
  case Register::Type::Integer:
    return static_cast<IntegralType>(std::forward<T>(value));
  case Register::Type::FloatingPoint:
    return static_cast<FloatingPointType>(std::forward<T>(value));
  case Register::Type::Bool:
    return static_cast<BooleanType>(std::forward<T>(value));
  case Register::Type::String:
    return static_cast<StringType>(std::forward<T>(value));
  }
}

bool isValidWindow(ElementStore &r, Register::Window &);

struct RegisterIterators {
  ElementStore::iterator rowBegin_;
  ElementStore::iterator rowEnd_;
  std::vector<Element>::iterator columnBegin_;
  std::vector<Element>::iterator columnEnd_;
};

struct Register::View::Impl {
  Register &r_;
  RegisterIterators it_;

  Impl(Register &r, Register::Window &w);
  RegisterIterators convertToIterators(Register &r, Window &w);
};

Register::View::Impl::Impl(Register &r, Register::Window &w)
    : r_(r), it_(convertToIterators(r, w)) {
  // todo:: move to chimeratk exceptions and revise error message to
  // something more helpful if possible.
  if (!isValidWindow(r.impl_->elementStore_, w)) {
    throw std::runtime_error("Window size is invalid for Register");
  }
}
Register::View::View(Register &r, Window w)
    : impl_(std::make_unique<Register::View::Impl>(r, w)) {}

bool isValidWindow(ElementStore &e, Register::Window &w) {
  auto rowSize = e.size();
  auto columnSize = e[0].size();
  Register::Shape &s = w.shape;
  auto lastRowIndex = s.getRows() + w.row_offset;
  auto lastColumnIndex = s.getColumns() + w.column_offset;
  return ((lastRowIndex < rowSize) && (lastColumnIndex < columnSize));
}

RegisterIterators
Register::View::Impl::convertToIterators(Register &r, Register::Window &w) {
  ElementStore &e = r.impl_->elementStore_;
  Register::Shape &s = w.shape;
  auto rowBegin = e.begin() + w.row_offset;
  auto rowEnd = rowBegin + s.getRows() + 1;
  auto columnBegin = e[0].begin();
  auto columnEnd = columnBegin + s.getColumns() + 1;
  return RegisterIterators //
      {
          rowBegin,    //
          rowEnd,      //
          columnBegin, //
          columnEnd    //
      };
}

template <typename UserType> //
std::vector<std::vector<UserType>> Register::View::read() {
  auto &l = impl_->it_;
  for (auto it = l.rowBegin_; it < l.rowEnd_; it++) {
    for (auto column = l.columnEnd_; column < l.columnEnd_; column++) {
    }
  }
}

template <typename UserType>
void Register::View::write(std::vector<std::vector<UserType>> &d) {
  auto &l = impl_->it_;
  for (auto it = l.rowBegin_; it < l.rowEnd_; it++) {
    for (auto column = l.columnEnd_; column < l.columnEnd_; column++) {
    }
  }
}

/*****************************************************************************/
// template specilizations
/*****************************************************************************/
template std::vector<std::vector<IntegralType>> //
pad(std::vector<std::vector<IntegralType>> v);
template std::vector<std::vector<FloatingPointType>> //
pad(std::vector<std::vector<FloatingPointType>> v);
template std::vector<std::vector<BooleanType>> //
pad(std::vector<std::vector<BooleanType>> v);
template std::vector<std::vector<StringType>> //
pad(std::vector<std::vector<StringType>> v);

template Register::Impl::Impl(std::string const &name, Access mode,
                              std::vector<std::vector<IntegralType>> data);
template Register::Impl::Impl(std::string const &name, Access mode,
                              std::vector<std::vector<FloatingPointType>> data);
template Register::Impl::Impl(std::string const &name, Access mode,
                              std::vector<std::vector<BooleanType>> data);
template Register::Impl::Impl(std::string const &name, Access mode,
                              std::vector<std::vector<StringType>> data);

template Register::Register(std::string const &name, Register::Access access,
                            std::vector<std::vector<IntegralType>> data);
template Register::Register(std::string const &name, Register::Access access,
                            std::vector<std::vector<FloatingPointType>> data);
template Register::Register(std::string const &name, Register::Access access,
                            std::vector<std::vector<BooleanType>> data);
template Register::Register(std::string const &name, Register::Access access,
                            std::vector<std::vector<StringType>> data);

template std::vector<std::vector<int8_t>> Register::read();
template std::vector<std::vector<int16_t>> Register::read();
template std::vector<std::vector<int32_t>> Register::read();
template std::vector<std::vector<int64_t>> Register::read();
template std::vector<std::vector<uint8_t>> Register::read();
template std::vector<std::vector<uint16_t>> Register::read();
template std::vector<std::vector<uint32_t>> Register::read();
template std::vector<std::vector<uint64_t>> Register::read();
template std::vector<std::vector<float>> Register::read();
template std::vector<std::vector<double>> Register::read();
template std::vector<std::vector<bool>> Register::read();
template std::vector<std::vector<std::string>> Register::read();

template void Register::write(std::vector<std::vector<int8_t>> const &data);
template void Register::write(std::vector<std::vector<int16_t>> const &data);
template void Register::write(std::vector<std::vector<int32_t>> const &data);
template void Register::write(std::vector<std::vector<int64_t>> const &data);
template void Register::write(std::vector<std::vector<uint8_t>> const &data);
template void Register::write(std::vector<std::vector<uint16_t>> const &data);
template void Register::write(std::vector<std::vector<uint32_t>> const &data);
template void Register::write(std::vector<std::vector<uint64_t>> const &data);
template void Register::write(std::vector<std::vector<float>> const &data);
template void Register::write(std::vector<std::vector<double>> const &data);
template void Register::write(std::vector<std::vector<bool>> const &data);
template void Register::write( //
    std::vector<std::vector<std::string>> const &data);
} // namespace TestBackend
