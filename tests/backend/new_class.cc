#include "new_class.h"
#include <boost/variant.hpp>
#include <iostream>

template <typename T> //
using DataContainer = std::vector<std::vector<T>>;
using Element = boost::variant<IntegralType,      //
                               FloatingPointType, //
                               BooleanType,       //
                               StringType>;
using ElementStore = DataContainer<Element>;

namespace TestBackend {

ElementStore buildElementStore(Register::Type t, Register::Shape s);
template <typename VariantType>
ElementStore convertToElementStore(DataContainer<VariantType> &d);
template <typename UserType>
Element convertToElement(Register::Type t, UserType &&value);
template <typename Type>
Register::Shape extractShape(const DataContainer<Type> &d);
template <typename UserType>
DataContainer<UserType> &pad(DataContainer<UserType> &input);
bool isValidWindow(ElementStore &r, Register::Window &);

/****************************************************************************/
template <typename UserType> class Converter {
public:
  template <typename VariantType> UserType operator()(VariantType &e) const {
    return static_cast<UserType>(e);
  }
};
struct RegisterIterators {
  ElementStore::iterator rowBegin_;
  ElementStore::iterator rowEnd_;
  std::vector<Element>::iterator columnBegin_;
  std::vector<Element>::iterator columnEnd_;
};
struct Register::Impl {
  std::string name_;
  Access access_;
  ElementStore elementStore_;

  Impl(std::string const &name, Access access, ElementStore e);
  static RegisterIterators getIterators(Register &r, Window &w);
};
struct Register::View::Impl {
  Register &r_;
  RegisterIterators it_;
  Impl(Register &r, RegisterIterators const &i);
};
/*****************************************************************************/
template <typename VariantType>
Register::Register(std::string const &name, //
                   Register::Access access, //
                   DataContainer<VariantType> data)
    : impl_(std::make_unique<Impl>(name, access,
                                   convertToElementStore(pad(data)))) {}

/*****************************************************************************/
Register::Register(std::string const &name, //
                   Access access,           //
                   Type type,               //
                   Shape shape)
    : impl_(std::make_unique<Impl>(name, access,
                                   buildElementStore(type, shape))) {}

/*****************************************************************************/
Register::Register(Register const &r)
    : impl_(std::make_unique<Impl>(r.impl_->name_,   //
                                   r.impl_->access_, //
                                   r.impl_->elementStore_)) {}

/*****************************************************************************/
Register::Register(Register &&r) : impl_(std::move(r.impl_)) {}

/*****************************************************************************/
Register::Impl::Impl(std::string const &name, Access access, ElementStore e)
    : name_(name), access_(access), elementStore_(e) {}

/*****************************************************************************/
Register::View::View(Register &r, Window w)
    : impl_(std::make_unique<Impl>(r, Register::Impl::getIterators(r, w))) {
  if (!isValidWindow(r.impl_->elementStore_, w)) {
    throw std::runtime_error("Window size is invalid for Register");
  }
}

/*****************************************************************************/
Register::View::Impl::Impl(Register &r, RegisterIterators const &i)
    : r_(r), it_(i) {}

/*****************************************************************************/
Register::View::View(Register::View const &r)
    : impl_(std::make_unique<Impl>(r.impl_->r_, r.impl_->it_)) {}

/*****************************************************************************/
Register::View::View(Register::View &&r) : impl_(std::move(r.impl_)) {}

/*****************************************************************************/
Register::Shape::Shape(size_t r, size_t c) : rows_(r), columns_(c) {
  if (rows_ == 0 || columns_ == 0) {
    throw std::logic_error("Invalid register shape requested; register must "
                           "accomodate at least one element");
  }
}
/*****************************************************************************/
bool Register::Shape::operator==(Shape const &rhs) {
  return (rows_ == rhs.rows_) && (columns_ == rhs.columns_);
}
/****************************************************************************/
bool Register::Shape::operator!=(Shape const &rhs) { return !operator==(rhs); }

/****************************************************************************/
// To get around the static assertion with unique ptr used for the pimpl
// pattern
Register::~Register() = default;
Register::View::~View() = default;

/****************************************************************************/
template <typename UserType> DataContainer<UserType> Register::read() {
  auto &elementStore = impl_->elementStore_;
  DataContainer<UserType> result;
  for (auto &row : elementStore) {
    result.emplace_back(std::vector<UserType>());
    for (auto &element : row) {
      result.back().push_back(boost::apply_visitor(Converter<UserType>(), //
                                                   element));
    }
  }
  return result;
}
/*****************************************************************************/
template <typename T> //
void Register::write(DataContainer<T> const &data) {
  auto &elementStore = impl_->elementStore_;
  auto registerType = getType();

  auto registerShape = extractShape(elementStore);
  auto dataShape = extractShape(data);
  if (registerShape != dataShape) {
    throw std::runtime_error(
        "Input container does not confirm to Register shape");
  }
  auto rowSize = registerShape.rowSize();
  auto columnSize = registerShape.columnSize();
  for (size_t row = 0; row < rowSize; row++) {
    for (size_t column = 0; column < columnSize; column++) {
      elementStore[row][column] =
          convertToElement(registerType, data[row][column]);
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

/****************************************************************************/
Register::View Register::getView(Window w) { return View{*this, w}; }

/****************************************************************************/
Register::Access Register::getAccessMode() { return impl_->access_; }

/****************************************************************************/
template <typename UserType>
Register::Shape extractShape(const DataContainer<UserType> &d) {
  return Register::Shape{d.size(), (d.size()) ? d[0].size() : 0};
}

/****************************************************************************/
RegisterIterators Register::Impl::getIterators(Register &r, Window &w) {
  ElementStore &e = r.impl_->elementStore_;
  Register::Shape &s = w.shape;
  auto rowBegin = e.begin() + w.row_offset;
  auto rowEnd = rowBegin + s.rowSize() + 1;
  auto columnBegin = e[0].begin();
  auto columnEnd = columnBegin + s.columnSize() + 1;
  return RegisterIterators //
      {
          rowBegin,    //
          rowEnd,      //
          columnBegin, //
          columnEnd    //
      };
}
/***************************************************************************/
template <typename UserType> //
DataContainer<UserType> Register::View::read() {
  DataContainer<UserType> result;
  auto &it = impl_->it_;
  for (auto row = it.rowBegin_; row < it.rowEnd_; row++) {
    result.emplace_back(std::vector<UserType>());
    for (auto column = it.columnBegin_; column < it.columnEnd_; column++) {
      result.back().push_back(
          boost::apply_visitor(Converter<UserType>(), *column));
    }
  }
}
/***************************************************************************/
template <typename UserType>
void Register::View::write(DataContainer<UserType> const &d) {
  auto &it = impl_->it_;
  auto &r = impl_->r_;
  for (auto row = it.rowBegin_; row < it.rowEnd_; row++) {
    auto i_r = 0;
    for (auto column = it.columnBegin_; column < it.columnEnd_; column++) {
      auto i_c = 0;
      *column = convertToElement(r.getType(), d[i_r][i_c]);
      i_c++;
    }
    i_r++;
  }
}
/****************************************************************************/
ElementStore buildElementStore(Register::Type t, Register::Shape s) {
  switch (t) {
  case Register::Type::Bool:
    return ElementStore(s.rowSize(), //
                        std::vector<Element>(s.columnSize(), BooleanType()));

  case Register::Type::FloatingPoint:
    return ElementStore(
        s.rowSize(), //
        std::vector<Element>(s.columnSize(), FloatingPointType()));

  case Register::Type::Integer:
    return ElementStore(s.rowSize(), //
                        std::vector<Element>(s.columnSize(), IntegralType()));

  case Register::Type::String:
    return ElementStore(
        s.rowSize(), //
        std::vector<Element>(s.columnSize(), Element{StringType()}));
  }
  return ElementStore();
}
/****************************************************************************/
template <typename VariantType>
DataContainer<VariantType> &pad(DataContainer<VariantType> &input) {
  // handle invalid shape 0*0
  if (input.size() == 0) {
    input.resize(1);
    input[0].resize(1);
    return input;
  }
  std::size_t maxRowSize = 0;
  for (auto const &row : input) {
    if (row.size() > maxRowSize) {
      maxRowSize = row.size();
    }
  }
  // hande invalid shape 1*0
  if (maxRowSize == 0) {
    maxRowSize = 1;
  }
  for (auto &row : input) {
    row.resize(maxRowSize);
  }
  return input;
}
/***************************************************************************/
template <typename UserType>
Element convertToElement(Register::Type t, UserType &&value) {
  switch (t) {
  case Register::Type::Integer:
    return static_cast<IntegralType>(std::forward<UserType>(value));
  case Register::Type::FloatingPoint:
    return static_cast<FloatingPointType>(std::forward<UserType>(value));
  case Register::Type::Bool:
    return static_cast<BooleanType>(std::forward<UserType>(value));
  case Register::Type::String:
    return static_cast<StringType>(std::forward<UserType>(value));
  }
}
/***************************************************************************/
bool isValidWindow(ElementStore &e, Register::Window &w) {
  auto rowSize = e.size();
  auto columnSize = e[0].size();
  Register::Shape &s = w.shape;
  auto lastRowIndex = (s.rowSize() - 1) + w.row_offset;
  auto lastColumnIndex = (s.columnSize() - 1) + w.column_offset;
  return ((lastRowIndex < rowSize) && (lastColumnIndex < columnSize));
}
/****************************************************************************/
template <typename VariantType>
ElementStore convertToElementStore(DataContainer<VariantType> &d) {
  ElementStore e;
  for (auto &row : d) {
    e.emplace_back(std::vector<Element>{});
    for (auto element : row) {
      e.back().emplace_back(Element(element));
    }
  }
  return e;
}
/****************************************************************************/
// template specilizations
/****************************************************************************/
template DataContainer<IntegralType> &pad(DataContainer<IntegralType> &v);
template DataContainer<FloatingPointType> &
pad(DataContainer<FloatingPointType> &v);
template DataContainer<BooleanType> &pad(DataContainer<BooleanType> &v);
template DataContainer<StringType> &pad(DataContainer<StringType> &v);

template Register::Register(std::string const &name, Register::Access access,
                            DataContainer<IntegralType> data);
template Register::Register(std::string const &name, Register::Access access,
                            DataContainer<FloatingPointType> data);
template Register::Register(std::string const &name, Register::Access access,
                            DataContainer<BooleanType> data);
template Register::Register(std::string const &name, Register::Access access,
                            DataContainer<StringType> data);

template DataContainer<int8_t> Register::read();
template DataContainer<int16_t> Register::read();
template DataContainer<int32_t> Register::read();
template DataContainer<int64_t> Register::read();
template DataContainer<uint8_t> Register::read();
template DataContainer<uint16_t> Register::read();
template DataContainer<uint32_t> Register::read();
template DataContainer<uint64_t> Register::read();
template DataContainer<float> Register::read();
template DataContainer<double> Register::read();
template DataContainer<bool> Register::read();
template DataContainer<std::string> Register::read();

template void Register::write(DataContainer<int8_t> const &data);
template void Register::write(DataContainer<int16_t> const &data);
template void Register::write(DataContainer<int32_t> const &data);
template void Register::write(DataContainer<int64_t> const &data);
template void Register::write(DataContainer<uint8_t> const &data);
template void Register::write(DataContainer<uint16_t> const &data);
template void Register::write(DataContainer<uint32_t> const &data);
template void Register::write(DataContainer<uint64_t> const &data);
template void Register::write(DataContainer<float> const &data);
template void Register::write(DataContainer<double> const &data);
template void Register::write(DataContainer<bool> const &data);
template void Register::write(DataContainer<std::string> const &data);

template DataContainer<int8_t> Register::View::read();
template DataContainer<int16_t> Register::View::read();
template DataContainer<int32_t> Register::View::read();
template DataContainer<int64_t> Register::View::read();
template DataContainer<uint8_t> Register::View::read();
template DataContainer<uint16_t> Register::View::read();
template DataContainer<uint32_t> Register::View::read();
template DataContainer<uint64_t> Register::View::read();
template DataContainer<float> Register::View::read();
template DataContainer<double> Register::View::read();
template DataContainer<bool> Register::View::read();
template DataContainer<std::string> Register::View::read();

template void Register::View::write(DataContainer<int8_t> const &d);
template void Register::View::write(DataContainer<int16_t> const &d);
template void Register::View::write(DataContainer<int32_t> const &d);
template void Register::View::write(DataContainer<int64_t> const &d);
template void Register::View::write(DataContainer<uint8_t> const &d);
template void Register::View::write(DataContainer<uint16_t> const &d);
template void Register::View::write(DataContainer<uint32_t> const &d);
template void Register::View::write(DataContainer<uint64_t> const &d);
template void Register::View::write(DataContainer<float> const &d);
template void Register::View::write(DataContainer<double> const &d);
template void Register::View::write(DataContainer<bool> const &d);
template void Register::View::write(DataContainer<std::string> const &d);
} // namespace TestBackend
