#include "register.h"

#include <ChimeraTK/SupportedUserTypes.h>
#include <ChimeraTK/Exception.h>
#include <boost/variant.hpp>
#include <iostream>

template<typename T> //
using DataContainer = std::vector<std::vector<T>>;
using Element = boost::variant<IntegralType, //
    FloatingPointType,                       //
    BooleanType,                             //
    StringType>;
using ElementStore = DataContainer<Element>;

namespace TestBackend {

  ElementStore buildElementStore(Register::Type t, Register::Shape s);
  template<typename VariantType>
  ElementStore convertToElementStore(DataContainer<VariantType>& d);
  template<typename UserType>
  Element convertToElement(Register::Type t, UserType&& value);
  template<typename Type>
  Register::Shape extractShape(const DataContainer<Type>& d);
  template<typename UserType>
  DataContainer<UserType>& pad(DataContainer<UserType>& input);
  bool isValidWindow(ElementStore& r, Register::Window&);

  /****************************************************************************/
  template<typename UserType>
  class Converter {
   public:
    template<typename VariantType>
    UserType operator()(VariantType& e) const {
      if constexpr(!std::is_same<UserType, ChimeraTK::Void>::value) {
        return static_cast<UserType>(e);
      }
      else {
        return {};
      }
    }
  };
  struct Indices {
    size_t rowStart;
    size_t rowLimit;
    size_t columnStart;
    size_t columnLimit;
  };
  struct Register::Impl {
    std::string name_;
    Access access_;
    ElementStore elementStore_;

    Impl(std::string const& name, Access access, ElementStore e);
  };
  struct Register::View::Impl {
    Register& r_;
    Indices i_;
    Impl(Register& r, Indices const& i);
  };
  Indices convertToIndices(ElementStore& e, Register::Window& w);

  std::string registerName(Register::View const& v) { return v.impl_->r_.getName(); }
  Register::Access getAccessMode(Register::View const& v) { return v.impl_->r_.getAccessMode(); }
  size_t columns(Register::View const& v) {
    auto columnStart = v.impl_->i_.columnStart;
    auto columnSize = v.impl_->i_.columnLimit;
    return (columnSize - columnStart);
  }
  size_t rows(Register::View const& v) {
    auto rowStart = v.impl_->i_.rowStart;
    auto rowSize = v.impl_->i_.rowLimit;
    return (rowSize - rowStart);
  }
  /*****************************************************************************/
  template<typename VariantType>
  Register::Register(std::string const& name, //
      Register::Access access,                //
      DataContainer<VariantType>
          data)
  : impl_(std::make_unique<Impl>(name, access, convertToElementStore(pad(data)))) {}

  /*****************************************************************************/
  Register::Register(std::string const& name, //
      Access access,                          //
      Type type,                              //
      Shape shape)
  : impl_(std::make_unique<Impl>(name, access, buildElementStore(type, shape))) {}

  /*****************************************************************************/
  Register::Register(Register const& r)
  : impl_(std::make_unique<Impl>(r.impl_->name_, //
        r.impl_->access_,                        //
        r.impl_->elementStore_)) {}

  /*****************************************************************************/
  Register::Register(Register&& r) : impl_(std::move(r.impl_)) {}

  /*****************************************************************************/
  Register::Impl::Impl(std::string const& name, Access access, ElementStore e)
  : name_(name), access_(access), elementStore_(e) {}

  /*****************************************************************************/
  Register::View::View(Register& r, Window w)
  : impl_(std::make_unique<Impl>(r, convertToIndices(r.impl_->elementStore_, w))) {
    if(!isValidWindow(r.impl_->elementStore_, w)) {
      throw ChimeraTK::runtime_error("Window size is invalid for Register");
    }
  }
  /*****************************************************************************/
  Register::View::View(Register::View const& r) : impl_(std::make_unique<Impl>(r.impl_->r_, r.impl_->i_)) {}

  /*****************************************************************************/
  Register::View::View(Register::View&& r) : impl_(std::move(r.impl_)) {}

  /*****************************************************************************/
  Indices convertToIndices(ElementStore& e, Register::Window& w) {
    return {w.row_offset, (w.row_offset + w.shape.rowSize()) < e.size() ? w.row_offset + w.shape.rowSize() : e.size(),
        w.column_offset,
        (w.column_offset + w.shape.columnSize()) < e[0].size() ? w.column_offset + w.shape.columnSize() : e[0].size()};
  }
  /*****************************************************************************/
  Register::View::Impl::Impl(Register& r, Indices const& i) : r_(r), i_(i) {}

  /*****************************************************************************/
  Register::Shape::Shape(size_t r, size_t c) : rows_(r), columns_(c) {
    if(rows_ == 0 || columns_ == 0) {
      throw ChimeraTK::logic_error("Invalid register shape requested; register must "
                                   "accomodate at least one element");
    }
  }
  /*****************************************************************************/
  bool Register::Shape::operator==(Shape const& rhs) { return (rows_ == rhs.rows_) && (columns_ == rhs.columns_); }
  /****************************************************************************/
  bool Register::Shape::operator!=(Shape const& rhs) { return !operator==(rhs); }

  /****************************************************************************/
  // To get around the static assertion with unique ptr used for the pimpl
  // pattern
  Register::~Register() = default;
  Register::View::~View() = default;

  /****************************************************************************/
  template<typename UserType>
  DataContainer<UserType> Register::read() {
    auto& elementStore = impl_->elementStore_;
    DataContainer<UserType> result;
    for(auto& row : elementStore) {
      result.emplace_back(std::vector<UserType>());
      for(auto& element : row) {
        result.back().push_back(boost::apply_visitor(Converter<UserType>(), //
            element));
      }
    }
    return result;
  }
  /*****************************************************************************/
  template<typename T> //
  void Register::write(DataContainer<T> const& data) {
    auto& elementStore = impl_->elementStore_;
    auto registerType = getType();

    auto registerShape = extractShape(elementStore);
    auto dataShape = extractShape(data);
    if(registerShape != dataShape) {
      throw ChimeraTK::runtime_error("Input container does not confirm to Register shape");
    }
    auto rowSize = registerShape.rowSize();
    auto columnSize = registerShape.columnSize();
    for(size_t row = 0; row < rowSize; row++) {
      for(size_t column = 0; column < columnSize; column++) {
        elementStore[row][column] = convertToElement(registerType, data[row][column]);
      }
    }
  }
  /*****************************************************************************/
  Register::Type Register::getType() const { return static_cast<Type>(impl_->elementStore_[0][0].which()); }
  /*****************************************************************************/
  Register::Shape Register::getShape() const { return extractShape(impl_->elementStore_); }
  /*****************************************************************************/
  std::string Register::getName() const { return impl_->name_; }

  /****************************************************************************/
  Register::View Register::getView(Window w) { return View{*this, w}; }

  /****************************************************************************/
  Register::Access Register::getAccessMode() const { return impl_->access_; }

  /****************************************************************************/
  template<typename UserType>
  Register::Shape extractShape(const DataContainer<UserType>& d) {
    return Register::Shape{d.size(), (d.size()) ? d[0].size() : 0};
  }
  /***************************************************************************/
  template<typename UserType> //
  DataContainer<UserType> Register::View::read() {
    DataContainer<UserType> result;
    auto& e = impl_->r_.impl_->elementStore_;
    auto& i = impl_->i_;
    for(auto r_index = i.rowStart; r_index < i.rowLimit; r_index++) {
      result.emplace_back(std::vector<UserType>());
      for(auto c_index = i.columnStart; c_index < i.columnLimit; c_index++) {
        result.back().push_back(boost::apply_visitor(Converter<UserType>(), //
            e[r_index][c_index]));
      }
    }
    return result;
  }
  /***************************************************************************/
  template<typename UserType>
  void Register::View::write(DataContainer<UserType> const& d) {
    auto& i = impl_->i_;
    auto& r = impl_->r_;
    auto& e = r.impl_->elementStore_;
    size_t i_r = 0;
    for(auto r_index = i.rowStart; r_index < i.rowLimit; r_index++) {
      size_t i_c = 0;
      for(auto c_index = i.columnStart; c_index < i.columnLimit; c_index++) {
        e[r_index][c_index] = convertToElement(r.getType(), d[i_r][i_c]);
        i_c++;
      }
      i_r++;
    }
  }
  /****************************************************************************/
  ElementStore buildElementStore(Register::Type t, Register::Shape s) {
    switch(t) {
      case Register::Type::Bool:
        return ElementStore(s.rowSize(), //
            std::vector<Element>(s.columnSize(), BooleanType()));

      case Register::Type::FloatingPoint:
        return ElementStore(s.rowSize(), //
            std::vector<Element>(s.columnSize(), FloatingPointType()));

      case Register::Type::Integer:
        return ElementStore(s.rowSize(), //
            std::vector<Element>(s.columnSize(), IntegralType()));

      case Register::Type::String:
        return ElementStore(s.rowSize(), //
            std::vector<Element>(s.columnSize(), Element{StringType()}));
    }
    return ElementStore();
  }
  /****************************************************************************/
  template<typename VariantType>
  DataContainer<VariantType>& pad(DataContainer<VariantType>& input) {
    // handle invalid shape 0*0
    if(input.size() == 0) {
      input.resize(1);
      input[0].resize(1);
      return input;
    }
    std::size_t maxRowSize = 0;
    for(auto const& row : input) {
      if(row.size() > maxRowSize) {
        maxRowSize = row.size();
      }
    }
    // hande invalid shape 1*0
    if(maxRowSize == 0) {
      maxRowSize = 1;
    }
    for(auto& row : input) {
      row.resize(maxRowSize);
    }
    return input;
  }
  /***************************************************************************/
  template<typename UserType>
  Element convertToElement(Register::Type t, UserType&& value) {
    if constexpr(!std::is_same<UserType, const ChimeraTK::Void&>::value) {
      switch(t) {
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
    // fixme:?;
    // we have covered all cases for Register::type; hence reaching here should
    // be incorrect.

    throw ChimeraTK::logic_error("incorrect type requested for conversion");
  }
  /***************************************************************************/
  template<>
  Element convertToElement<const ChimeraTK::Void&>(Register::Type, const ChimeraTK::Void&) {
    throw ChimeraTK::logic_error("incorrect type requested for conversion");
  }
  /***************************************************************************/
  bool isValidWindow(ElementStore& e, Register::Window& w) {
    auto rowSize = e.size();
    auto columnSize = e[0].size();
    Register::Shape& s = w.shape;
    auto lastRowIndex = (s.rowSize() - 1) + w.row_offset;
    auto lastColumnIndex = (s.columnSize() - 1) + w.column_offset;
    return ((lastRowIndex < rowSize) && (lastColumnIndex < columnSize));
  }
  /****************************************************************************/
  template<typename VariantType>
  ElementStore convertToElementStore(DataContainer<VariantType>& d) {
    ElementStore e;
    for(auto& row : d) {
      e.emplace_back(std::vector<Element>{});
      for(auto element : row) {
        e.back().emplace_back(Element(element));
      }
    }
    return e;
  }
  /****************************************************************************/
  size_t columns(Register const& r) { return r.getShape().columnSize(); }

  /****************************************************************************/
  size_t rows(Register const& r) { return r.getShape().rowSize(); }

  /****************************************************************************/
  // template instantiations
  /****************************************************************************/
  template DataContainer<IntegralType>& pad(DataContainer<IntegralType>& v);
  template DataContainer<FloatingPointType>& pad(DataContainer<FloatingPointType>& v);
  template DataContainer<BooleanType>& pad(DataContainer<BooleanType>& v);
  template DataContainer<StringType>& pad(DataContainer<StringType>& v);

  template Register::Register(std::string const& name, Register::Access access, DataContainer<IntegralType> data);
  template Register::Register(std::string const& name, Register::Access access, DataContainer<FloatingPointType> data);
  template Register::Register(std::string const& name, Register::Access access, DataContainer<BooleanType> data);
  template Register::Register(std::string const& name, Register::Access access, DataContainer<StringType> data);

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
  template DataContainer<std::string> Register::read();
  template DataContainer<ChimeraTK::Boolean> Register::read();
  template DataContainer<ChimeraTK::Void> Register::read();

  template void Register::write(DataContainer<int8_t> const& data);
  template void Register::write(DataContainer<int16_t> const& data);
  template void Register::write(DataContainer<int32_t> const& data);
  template void Register::write(DataContainer<int64_t> const& data);
  template void Register::write(DataContainer<uint8_t> const& data);
  template void Register::write(DataContainer<uint16_t> const& data);
  template void Register::write(DataContainer<uint32_t> const& data);
  template void Register::write(DataContainer<uint64_t> const& data);
  template void Register::write(DataContainer<float> const& data);
  template void Register::write(DataContainer<double> const& data);
  template void Register::write(DataContainer<std::string> const& data);
  template void Register::write(DataContainer<ChimeraTK::Boolean> const& data);
  template void Register::write(DataContainer<ChimeraTK::Void> const& data);

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
  template DataContainer<std::string> Register::View::read();
  template DataContainer<ChimeraTK::Boolean> Register::View::read();
  template DataContainer<ChimeraTK::Void> Register::View::read();

  template void Register::View::write(DataContainer<int8_t> const& d);
  template void Register::View::write(DataContainer<int16_t> const& d);
  template void Register::View::write(DataContainer<int32_t> const& d);
  template void Register::View::write(DataContainer<int64_t> const& d);
  template void Register::View::write(DataContainer<uint8_t> const& d);
  template void Register::View::write(DataContainer<uint16_t> const& d);
  template void Register::View::write(DataContainer<uint32_t> const& d);
  template void Register::View::write(DataContainer<uint64_t> const& d);
  template void Register::View::write(DataContainer<float> const& d);
  template void Register::View::write(DataContainer<double> const& d);
  template void Register::View::write(DataContainer<std::string> const& d);
  template void Register::View::write(DataContainer<ChimeraTK::Boolean> const& d);
  template void Register::View::write(DataContainer<ChimeraTK::Void> const& d);
} // namespace TestBackend
