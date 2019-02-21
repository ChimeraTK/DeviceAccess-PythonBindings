/*
 * VariantTypes.h
 *
 *  Created on: Sep 6, 2018
 *      Author: varghese
 */

#ifndef TESTS_BACKEND_VARIANTTYPES_H_
#define TESTS_BACKEND_VARIANTTYPES_H_

#include <ChimeraTK/Exception.h>
#include <cstdint>
#include <stdexcept>

/*
 * wrappers to support:
 *   static_cast<UserType>(VariantType)
 *   static_cast<VariantType>(UserType)
 *
 *   UserType:
 *     int8/16/32/64
 *     uint8/16/32/64
 *     float32/64
 *     std::string
 *   VariantType:
 *     Integral
 *     FloatingPoint
 *     Boolean
 *     std::string
 */
class IntegralType {
  int64_t value{0};

public:
  IntegralType() = default;
  template <typename T> IntegralType(T const &numeric) : value(numeric) {}
  operator auto() const { return value; }

  IntegralType(std::string) {
    throw ChimeraTK::logic_error("invalid conversion: string -> integer");
  }
  operator std::string() const {
    throw ChimeraTK::logic_error("invalid conversion: integer - > string");
  }
};

class FloatingPointType {
  double value{0};

public:
  FloatingPointType() = default;
  template <typename T> FloatingPointType(T const &numeric) : value(numeric) {}
  operator auto() const { return value; }

  FloatingPointType(std::string) {
    throw ChimeraTK::logic_error(
        "invalid conversion: string - > floatingPoint");
  }
  operator std::string() const {
    throw ChimeraTK::logic_error(
        "invalid conversion: floatingPoint - > string");
  }
};

class BooleanType {
  bool value{false};

public:
  BooleanType() = default;
  template <typename T> BooleanType(T const &numeric) : value(numeric) {}
  operator auto() const { return value; }

  BooleanType(std::string) {
    throw ChimeraTK::logic_error("invalid conversion: string - > bool");
  }
  operator std::string() const {
    throw ChimeraTK::logic_error("invalid conversion: bool - > string");
  }
};

class StringType {
  std::string value{""};

public:
  StringType() = default;
  StringType(std::string const &s) : value(s) {}

  operator std::string() const { return value; }

  template <typename T> StringType(T) {
    throw ChimeraTK::logic_error("invalid conversion to string");
  }
  template <typename T,
            typename = std::enable_if_t<std::is_floating_point<T>::value ||
                                        std::is_integral<T>::value>>
  operator T() const {
    throw ChimeraTK::logic_error("invalid conversion from string type");
  }
};
#endif /* TESTS_BACKEND_VARIANTTYPES_H_ */
