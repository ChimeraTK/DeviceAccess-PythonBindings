/*
 * VariantTypes.h
 *
 *  Created on: Sep 6, 2018
 *      Author: varghese
 */

#ifndef TESTS_BACKEND_VARIANTTYPES_H_
#define TESTS_BACKEND_VARIANTTYPES_H_

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
  int64_t value{};

public:
  IntegralType() = default;
  template <typename T> IntegralType(T numeric) : value(numeric) {}
  operator auto() { return value; }

  IntegralType(std::string) {
    throw std::logic_error("invalid conversion: string -> integer");
  }
  operator std::string() {
    throw std::logic_error("invalid conversion: integer - > string");
  }
};

class FloatingPointType {
  double value{};

public:
  FloatingPointType() = default;
  template <typename T> FloatingPointType(T numeric) : value(numeric){};
  operator auto() { return value; }

  FloatingPointType(std::string) {
    throw std::logic_error("invalid conversion: string - > floatingPoint");
  }
  operator std::string() {
    throw std::logic_error("invalid conversion: floatingPoint - > string");
  }
};

class BooleanType {
  bool value;

public:
  BooleanType() = default;
  template <typename T> BooleanType(T numeric) : value(numeric) {}
  operator auto() { return value; }

  BooleanType(std::string) {
    throw std::logic_error("invalid conversion: string - > bool");
  }
  operator std::string() {
    throw std::logic_error("invalid conversion: bool - > string");
  }
};

class StringType {
  std::string value{};

public:
  StringType() = default;
  StringType(std::string s) : value(std::move(s)) {}
  operator std::string() { return value; }

  template <typename T> StringType(T) {
    throw std::logic_error("invalid conversion to string");
  }
  operator auto() {
    throw std::logic_error("invalid conversion from string type");
  }
};
#endif /* TESTS_BACKEND_VARIANTTYPES_H_ */
