#ifndef REGISTER_ACCESS_H
#define REGISTER_ACCESS_H

#include <string>
#include <vector>
#include <iterator>
#include <memory>
#include <tuple>
#include <set>
#include <boost/variant.hpp>

#include <string>

namespace TestBackend {

enum class ElementType;
class DBaseElem;
enum class AccessMode;

template <typename T>
std::vector<std::vector<T> > pad(std::vector<std::vector<T> > v);

template <typename UserType>
std::vector<std::vector<UserType> >    //
    copyAs(const DBaseElem& e,         //
           std::size_t x_elements = 0, //
           std::size_t y_elements = 0, //
           std::size_t x_offset = 0,   //
           std::size_t y_offset = 0);

template <typename UserType>
std::tuple<std::size_t, std::size_t>                   //
    copyFrom(const DBaseElem& e,                       //
             std::vector<std::vector<UserType> >& out, //
             std::size_t x_offset = 0,                 //
             std::size_t y_offset = 0);

template <typename UserType>
std::tuple<std::size_t, std::size_t>                          //
    copyInto(DBaseElem& e,                                    //
             std::vector<std::vector<UserType> > const& data, //
             std::size_t x_offset = 0, std::size_t y_offset = 0);
bool isSubShape(DBaseElem const& e,   //
                std::size_t x_size,   //
                std::size_t y_size,   //
                std::size_t x_offset, //
                std::size_t y_offset);

std::tuple<std::size_t, std::size_t> //
    shape(const DBaseElem& e);

ElementType type(DBaseElem const& e);
std::string id(DBaseElem const& e);

AccessMode access(DBaseElem const& e);

/*============================================================================*/
enum class AccessMode {
  rw,
  ro,
  wo
};

enum class ElementType {
  Int,
  Double,
  String,
  Bool
};

struct Int_t {
  Int_t(int a) : data_(a) {}
  Int_t() : data_() {}
  Int_t(std::string const &/*s*/): data_(){
    //TODO: replace with chimeratk exception?
    throw std::runtime_error("conversion from string to int not supported");
  }

  operator double() const { return data_; }

  operator bool() const {
    if (data_) {
      return true;
    } else {
      return false;
    }
  }

  operator int8_t() const { return data_; }
  operator uint8_t() const { return data_; }
  operator int16_t() const { return data_; }
  operator uint16_t() const { return data_; }
  operator uint32_t() const { return data_; }
  operator int32_t() const { return data_; }
  operator int64_t() const { return data_; }
  operator uint64_t() const { return data_; }
  operator float() const { return data_; }
  operator std::string() const { return std::to_string(data_); }

private:
  int data_;
};

struct Double_t {
  Double_t(double a) : data_(a) {}
  Double_t() : data_() {}
  Double_t(std::string const &/*s*/): data_(){
    //TODO: replace with chimeratk exception?
    throw std::runtime_error("conversion from string to double not supported");
  }

  operator double() const { return data_; }

  operator bool() const {
    if (data_) {
      return true;
    } else {
      return false;
    }
  }

  operator int8_t() const { return data_; }
  operator uint8_t() const { return data_; }
  operator int16_t() const { return data_; }
  operator uint16_t() const { return data_; }
  operator int32_t() const { return data_; }
  operator uint32_t() const { return data_; }
  operator int64_t() const { return data_; }
  operator uint64_t() const { return data_; }
  operator float() const { return data_; }
  operator std::string() const { return std::to_string(data_); }

private:
  double data_;
};

struct Bool_t {

  Bool_t(bool v) : data_(v) {};
  Bool_t() : data_(false) {}
  Bool_t(std::string const & /*s*/): data_(){
    //TODO: replace with chimeratk exception?
    throw std::runtime_error("conversion from string to Bool not supported");
  }

  operator double() const { return data_; }
  operator bool() const { return data_; }
  operator int8_t() const { return data_; }
  operator uint8_t() const { return data_; }
  operator int16_t() const { return data_; }
  operator uint16_t() const { return data_; }
  operator int32_t() const { return data_; }
  operator uint32_t() const { return data_; }
  operator int64_t() const { return data_; }
  operator uint64_t() const { return data_; }
  operator float() const { return data_; }

  operator std::string() const {
    if (data_) {
      return std::string("true");
    } else {
      return std::string("false");
    }
  }

private:
  bool data_;
};

struct String_t {

  String_t(std::string const& string) : data_(string) {}
  String_t(const char* string) : data_(string) {}
  String_t() : data_() {}

  // intended to support static cast from numeric usertypes:
  String_t(int /*data*/) : data_() {
    // TODO: conversion error
    throw std::runtime_error(
        "Invalid conversion from numeric types to string requested");
  }

  operator double() const {
    // TODO: replace with the releavent chimeratk exception
    throw std::runtime_error(
        "Invalid conversion from string type to double requested");
  }

  operator bool() const {
    // TODO: replace with the releavent chimeratk exception
    throw std::runtime_error(
        "Invalid conversion from string type to bool requested");
  }

  operator int8_t() const {
    // TODO: replace with the releavent chimeratk exception
    throw std::runtime_error(
        "Invalid conversion from string type to int8_t requested");
  }

  operator uint8_t() const {
    // TODO: replace with the releavent chimeratk exception
    throw std::runtime_error(
        "Invalid conversion from string type to uint8_t requested");
  }

  operator int16_t() const {
    // TODO: replace with the releavent chimeratk exception
    throw std::runtime_error(
        "Invalid conversion from string type to int16_t requested");
  }

  operator uint16_t() const {
    // TODO: replace with the releavent chimeratk exception
    throw std::runtime_error(
        "Invalid conversion from string type to uint16_t requested");
  }

  operator int32_t() const {
    // TODO: replace with the releavent chimeratk exception
    throw std::runtime_error(
        "Invalid conversion from string type to int32_t requested");
  }

  operator uint32_t() const {
    // TODO: replace with the releavent chimeratk exception
    throw std::runtime_error(
        "Invalid conversion from string type to uint32_t requested");
  }

  operator int64_t() const {
    // TODO: replace with the releavent chimeratk exception
    throw std::runtime_error(
        "Invalid conversion from string type to int64_t requested");
  }

  operator uint64_t() const {
    // TODO: replace with the releavent chimeratk exception
    throw std::runtime_error(
        "Invalid conversion from string type to uint64_t requested");
  }

  operator float() const {
    // TODO: replace with the releavent chimeratk exception
    throw std::runtime_error(
        "Invalid conversion from string type to float requested");
  }

  operator std::string() const { return data_; }

  String_t(String_t const& rhs) = default;
  String_t(String_t&& rhs) = default;
  String_t& operator=(String_t const& rhs) = default;
  String_t& operator=(String_t&& rhs) = default;

private:
  std::string data_;
};

// type
// name/id
// access
class DBaseElem {
public:
  using Value_t = boost::variant<std::vector<std::vector<Int_t> >,    //
                                 std::vector<std::vector<Double_t> >, //
                                 std::vector<std::vector<String_t> >, //
                                 std::vector<std::vector<Bool_t> > >;
  template <typename T> using Container_t = std::vector<std::vector<T> >;

  template <typename Data_t>
  DBaseElem(std::string const& name,          //
            Container_t<Data_t> const& value, //
            AccessMode access = AccessMode::rw)
      : name_(name), value_(pad(value)), access_(access) {}

  template <typename UserType>
  friend std::vector<std::vector<UserType> > //
      copyAs(const DBaseElem& e,             //
             std::size_t x_elements,         //
             std::size_t y_elements,         //
             std::size_t x_offset,           //
             std::size_t y_offset);

  template <typename UserType>
  friend std::tuple<std::size_t, std::size_t>            //
      copyFrom(const DBaseElem& e,                       //
               std::vector<std::vector<UserType> >& out, //
               std::size_t x_offset,                     //
               std::size_t y_offset);

  template <typename UserType>
  friend std::tuple<std::size_t, std::size_t>                   //
      copyInto(DBaseElem& e,                                    //
               std::vector<std::vector<UserType> > const& data, //
               std::size_t x_offset,                            //
               std::size_t y_offset);

  friend std::tuple<std::size_t, std::size_t> //
      shape(const DBaseElem& e);
  friend ElementType type(DBaseElem const& e);
  friend std::string id(DBaseElem const& e);
  friend AccessMode access(DBaseElem const& e);

  std::size_t getChannels() const;
  std::size_t getElements() const;
  std::size_t getDimensions() const;

  DBaseElem(const DBaseElem& rhs) = default;
  DBaseElem& operator=(const DBaseElem& rhs) = default;
  DBaseElem& operator=(DBaseElem&& rhs) = default;
  DBaseElem(DBaseElem&& rhs) = default;

  bool operator<(DBaseElem const& rhs) const { return (name_ < rhs.name_); }
  bool operator<(std::string const& rhs) const { return (name_ < rhs); }
  bool operator>(DBaseElem const& rhs) const { return (name_ > rhs.name_); }
  bool operator==(std::string const& s) const { return (name_ == s); }

private:
  std::string name_;
  Value_t value_;
  AccessMode access_;
};

// Variant Visitors
/*============================================================================*/
template <typename UserType> struct CopyAs {

  CopyAs(std::size_t x_offset,   //
         std::size_t x_elements, //
         std::size_t y_offset,   //
         std::size_t y_elements)
      : x_offset_(x_offset),
        x_elements_(x_elements),
        y_offset_(y_offset),
        y_elements_(y_elements) {}

  using Container_t = std::vector<std::vector<UserType> >;

  template <typename VariantType>
  Container_t operator()(VariantType const& container) {

    Container_t tmp;
    if (container.size() == 0) {
      return tmp;
    }

    if (x_elements_ == 0) {
      x_elements_ = container[0].size();
    }
    if (y_elements_ == 0) {
      y_elements_ = container.size();
    }

    auto y_begin = container.begin() + y_offset_;
    auto y_end = ((y_begin + y_elements_) > container.end())
                     ? container.end()
                     : y_begin + y_elements_;
    auto x_begin = container[0].begin() + x_offset_;
    auto x_end = ((x_begin + x_elements_) > container[0].end())
                     ? container[0].end()
                     : x_begin + x_elements_;

    for (auto y_it = y_begin; y_it < y_end; ++y_it) {
      std::vector<UserType> tmp_inner;
      for (auto x_it = x_begin; x_it < x_end; ++x_it) {
        tmp_inner.push_back(static_cast<UserType>(*x_it));
      }
      tmp.emplace_back(std::move(tmp_inner));
    }

    return tmp;
  }

private:
  std::size_t x_offset_;
  std::size_t x_elements_;
  std::size_t y_offset_;
  std::size_t y_elements_;
};

template <typename UserType> struct CopyFrom {

  using Container_t = std::vector<std::vector<UserType> >;

  CopyFrom(Container_t& out,     //
           std::size_t x_offset, //
           std::size_t y_offset)
      : out_(out), //
        x_offset_(x_offset),
        y_offset_(y_offset) {}

  template <typename VariantType>
  std::tuple<std::size_t, std::size_t> operator()(VariantType const& c) {

    std::size_t x_count = 0;
    std::size_t y_count = 0;

    if (c.size() == 0 || out_.size() == 0) {
      return std::make_tuple(x_count, y_count);
    }

    auto y_size = out_.size();
    auto x_size = out_[0].size();

    auto y_begin = c.begin() + y_offset_;
    auto y_end = (y_begin + y_size > c.end()) ? c.end() : y_begin + y_size;

    auto x_begin = c[0].begin() + x_offset_;
    auto x_end =
        (x_begin + x_size > c[0].end()) ? c[0].end() : x_begin + x_size;

    for (auto y_it = y_begin; y_it < y_end; ++y_it, ++y_count) {
      x_count = 0;
      for (auto x_it = x_begin; x_it < x_end; ++x_it, ++x_count) {
        out_[y_count][x_count] = static_cast<UserType>(*x_it);
      }
    }
    return std::make_tuple(x_count, y_count);
  }

private:
  Container_t& out_;
  std::size_t x_offset_;
  std::size_t y_offset_;
};

template <typename UserType> class Put {
  using Container_t = std::vector<std::vector<UserType> >;
  Container_t const& in_;
  std::size_t x_offset_;
  std::size_t y_offset_;

public:
  Put(Container_t const& in, std::size_t x_offset, std::size_t y_offset)
      : in_(in), x_offset_(x_offset), y_offset_(y_offset) {}

  template <typename T>
  std::tuple<std::size_t, std::size_t> operator()(
      std::vector<std::vector<T> >& c) {
    std::size_t x_count = 0;
    std::size_t y_count = 0;

    if (c.size() == 0 || in_.size() == 0) {
      return std::make_tuple(x_count, y_count);
    }

    auto y_size = in_.size();
    auto x_size = in_[0].size();

    auto y_begin = c.begin() + y_offset_;
    auto y_end = (y_begin + y_size > c.end()) ? c.end() : y_begin + y_size;

    auto x_begin = c[0].begin() + x_offset_;
    auto x_end =
        (x_begin + x_size > c[0].end()) ? c[0].end() : x_begin + x_size;

    for (auto y_it = y_begin; y_it < y_end; ++y_it, ++y_count) {
      x_count = 0;
      for (auto x_it = x_begin; x_it < x_end; ++x_it, ++x_count) {
        *x_it = static_cast<T>(in_[y_count][x_count]);
      }
    }
    return std::make_tuple(x_count, y_count);
  }
};

struct GetType {
  ElementType operator()(std::vector<std::vector<Int_t> > const& /*v*/) {
    return ElementType::Int;
  }
  ElementType operator()(std::vector<std::vector<Double_t> > const& /*v*/) {
    return ElementType::Double;
  }
  ElementType operator()(std::vector<std::vector<String_t> > const& /*v*/) {
    return ElementType::String;
  }
  ElementType operator()(std::vector<std::vector<Bool_t> > const& /*v*/) {
    return ElementType::Bool;
  }
};

struct GetChannels {
  template <typename T> std::size_t operator()(std::vector<std::vector<T> > const &v) {
    return v.size();
  }
};

struct GetSequences {
  template <typename T> std::size_t operator()(std::vector<std::vector<T> > const & v) {
    if (v.size() != 0) {
      return v[0].size();
    } else {
      return 0;
    }
  }
};

struct GetShape {
  template <typename T>
  std::tuple<std::size_t, std::size_t> operator()(
      std::vector<std::vector<T> > c) {

    if (c.size() == 0) {
      return std::make_tuple(0, 0);
    }
    auto y_size = c.size();
    auto x_size = c[0].size(); // FIXME: what if this is 0 and y_size aint
    return std::make_tuple(x_size, y_size);
  }
};

/*============================================================================*/
template <typename UserType>
std::vector<std::vector<UserType> > //
    copyAs(const DBaseElem& e,      //
           std::size_t x_elements,  //
           std::size_t y_elements,  //
           std::size_t x_offset,    //
           std::size_t y_offset) {

  if (e.access_ == AccessMode::wo) {
    // TODO: move to chimeratk exceptions
    throw std::runtime_error("Attempted read on write only resource");
  }

  auto visitor = CopyAs<UserType>(x_offset,   //
                                  x_elements, //
                                  y_offset,   //
                                  y_elements);

  return boost::apply_visitor(visitor, e.value_);
}
template <typename UserType>
std::tuple<std::size_t, std::size_t>                   //
    copyFrom(const DBaseElem& e,                       //
             std::vector<std::vector<UserType> >& out, //
             std::size_t x_offset,                     //
             std::size_t y_offset) {

  if (e.access_ == AccessMode::wo) {
    // TODO: move to chimeratk exceptions
    throw std::runtime_error("Attempted read on write only resource");
  }

  auto visitor = CopyFrom<UserType>(out, x_offset, y_offset);
  return boost::apply_visitor(visitor, e.value_);
}

template <typename T>
std::vector<std::vector<T> > pad(std::vector<std::vector<T> > v) {
  std::size_t x_max = 0;
  for (auto const& row : v) {
    if (row.size() > x_max) {
      x_max = row.size();
    }
  }
  for (auto& row : v) {
    row.resize(x_max);
  }

  return std::move(v);
}

template <typename UserType>
std::tuple<std::size_t, std::size_t>                          //
    copyInto(DBaseElem& e,                                    //
             std::vector<std::vector<UserType> > const& data, //
             std::size_t x_offset, std::size_t y_offset) {

  if (e.access_ == AccessMode::ro) {
    // TODO: move to chimeratk exceptions
    throw std::runtime_error("Attempted write on read only resource");
  }

  auto visitor = Put<UserType>(data, x_offset, y_offset);
  return boost::apply_visitor(visitor, e.value_);
}

} // namespace TestBackend

#endif
