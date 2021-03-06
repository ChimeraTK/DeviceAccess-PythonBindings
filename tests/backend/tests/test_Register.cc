﻿#define BOOST_TEST_MODULE testRegister
#include <boost/test/included/unit_test.hpp>
#include <boost/variant.hpp>

#include "register.h"
#include "variant_types.h"

// example usage:
// convertTo<int>(floatArray)
template<typename To, typename From>
std::vector<std::vector<To>> convertTo(std::vector<std::vector<From>> const& i);
template<typename T>
std::vector<std::vector<T>> slice(std::vector<std::vector<T>> const& i, TestBackend::Register::Window const& w);

BOOST_AUTO_TEST_CASE(testRegisterCreation) {
  // check for invalid shape
  BOOST_CHECK_THROW(
      (TestBackend::Register{"", TestBackend::Register::Access::rw, TestBackend::Register::Type::Integer, {1, 0}}),
      ChimeraTK::logic_error);
  BOOST_CHECK_THROW(
      (TestBackend::Register{"", TestBackend::Register::Access::rw, TestBackend::Register::Type::Integer, {0, 1}}),
      ChimeraTK::logic_error);
  BOOST_CHECK_THROW(
      (TestBackend::Register{"", TestBackend::Register::Access::rw, TestBackend::Register::Type::Integer, {0, 0}}),
      ChimeraTK::logic_error);
  BOOST_CHECK_NO_THROW(
      (TestBackend::Register{"", TestBackend::Register::Access::rw, TestBackend::Register::Type::Integer, {1, 1}}));
  BOOST_CHECK_NO_THROW(
      (TestBackend::Register{"", TestBackend::Register::Access::rw, std::vector<std::vector<IntegralType>>{{}}}

          ));
  BOOST_CHECK_NO_THROW(
      (TestBackend::Register{"", TestBackend::Register::Access::rw, std::vector<std::vector<FloatingPointType>>{{}}}

          ));
  BOOST_CHECK_NO_THROW(
      (TestBackend::Register{"", TestBackend::Register::Access::rw, std::vector<std::vector<BooleanType>>{{}}}

          ));
  BOOST_CHECK_NO_THROW(
      (TestBackend::Register{"", TestBackend::Register::Access::rw, std::vector<std::vector<StringType>>{{}}}

          ));
}
BOOST_AUTO_TEST_CASE(testRegisterRead) {
  auto values = std::vector<std::vector<IntegralType>>{{1, 3, 5}};
  TestBackend::Register e{"integral",    //
      TestBackend::Register::Access::rw, //
      values};

  auto expected = convertTo<int>(values);
  auto data = e.read<int>();
  BOOST_CHECK(expected == data);
}
BOOST_AUTO_TEST_CASE(testRegisterWrite) {
  TestBackend::Register r{"integral", TestBackend::Register::Access::rw, TestBackend::Register::Type::Integer, {1, 3}};
  auto expected = std::vector<std::vector<int>>{{2, 3, 5}};
  r.write(expected);
  auto data = r.read<int>();

  BOOST_CHECK(expected == data);
  BOOST_CHECK_THROW(r.write(std::vector<std::vector<int>>{{1, 2, 3, 4, 5}}), ChimeraTK::runtime_error);
}
BOOST_AUTO_TEST_CASE(testGetShape) {
  auto r = TestBackend::Register{"", TestBackend::Register::Access::rw, TestBackend::Register::Type::Integer, {4, 4}};
  BOOST_CHECK((r.getShape() == TestBackend::Register::Shape{4, 4}));
}

struct FixtureContent {
  static const unsigned int SIZE = 4;
  using ListOfNames = std::array<std::string, SIZE>;
  using ListOfRegisterTypes = std::array<TestBackend::Register::Type, SIZE>;
  using ListOfAccessTypes = std::array<TestBackend::Register::Access, SIZE>;

  ListOfNames names //
      {{
          "IntegerRegister",       //
          "FloatingPointRegister", //
          "BooleanRegister",       //
          "StringRegister"         //
      }};
  ListOfRegisterTypes types //
      {{
          TestBackend::Register::Type::Integer,       //
          TestBackend::Register::Type::FloatingPoint, //
          TestBackend::Register::Type::Bool,          //
          TestBackend::Register::Type::String         //
      }};
  ListOfAccessTypes access //
      {{
          TestBackend::Register::Access::rw, //
          TestBackend::Register::Access::ro, //
          TestBackend::Register::Access::wo, //
      }};
  std::vector<TestBackend::Register> registerList;

  FixtureContent() {
    for(unsigned int i = 0; i < SIZE; i++) {
      registerList.emplace_back(names[i], access[i], types[i]);
    }
  }
};

BOOST_FIXTURE_TEST_SUITE(registerTests, FixtureContent)

BOOST_AUTO_TEST_CASE(testNumericTypeConversions) {
  std::vector<TestBackend::Register> numeric;

  std::copy_if(registerList.begin(), //
      registerList.end(),            //
      std::back_inserter(numeric),
      [](TestBackend::Register& r) { return r.getType() != TestBackend::Register::Type::String; });

  auto runTests = [](TestBackend::Register& r) {
    BOOST_CHECK_NO_THROW(r.write<int8_t>({{-2}}));
    BOOST_CHECK_NO_THROW(r.write<int16_t>({{-2}}));
    BOOST_CHECK_NO_THROW(r.write<int32_t>({{-2}}));
    BOOST_CHECK_NO_THROW(r.write<int64_t>({{-2}}));
    BOOST_CHECK_NO_THROW(r.write<uint8_t>({{2}}));
    BOOST_CHECK_NO_THROW(r.write<uint16_t>({{2}}));
    BOOST_CHECK_NO_THROW(r.write<uint32_t>({{2}}));
    BOOST_CHECK_NO_THROW(r.write<uint64_t>({{2}}));
    BOOST_CHECK_NO_THROW(r.write<float>({{2.0}}));
    BOOST_CHECK_NO_THROW(r.write<double>({{2.7643}}));
    BOOST_CHECK_NO_THROW(r.write<bool>({{false}}));
    BOOST_CHECK_THROW(r.write<std::string>({{"test"}}), ChimeraTK::logic_error);

    BOOST_CHECK_NO_THROW(r.read<int8_t>());
    BOOST_CHECK_NO_THROW(r.read<int16_t>());
    BOOST_CHECK_NO_THROW(r.read<int32_t>());
    BOOST_CHECK_NO_THROW(r.read<int64_t>());
    BOOST_CHECK_NO_THROW(r.read<uint8_t>());
    BOOST_CHECK_NO_THROW(r.read<uint16_t>());
    BOOST_CHECK_NO_THROW(r.read<uint32_t>());
    BOOST_CHECK_NO_THROW(r.read<uint64_t>());
    BOOST_CHECK_NO_THROW(r.read<float>());
    BOOST_CHECK_NO_THROW(r.read<double>());
    BOOST_CHECK_NO_THROW(r.read<bool>());
    BOOST_CHECK_THROW(r.read<std::string>(), ChimeraTK::logic_error);

    auto v = r.getView({r.getShape(), 0, 0});
    BOOST_CHECK_NO_THROW(v.write<int8_t>({{-2}}));
    BOOST_CHECK_NO_THROW(v.write<int16_t>({{-2}}));
    BOOST_CHECK_NO_THROW(v.write<int32_t>({{-2}}));
    BOOST_CHECK_NO_THROW(v.write<int64_t>({{-2}}));
    BOOST_CHECK_NO_THROW(v.write<uint8_t>({{2}}));
    BOOST_CHECK_NO_THROW(v.write<uint16_t>({{2}}));
    BOOST_CHECK_NO_THROW(v.write<uint32_t>({{2}}));
    BOOST_CHECK_NO_THROW(v.write<uint64_t>({{2}}));
    BOOST_CHECK_NO_THROW(v.write<float>({{2.0}}));
    BOOST_CHECK_NO_THROW(v.write<double>({{2.7643}}));
    BOOST_CHECK_NO_THROW(v.write<bool>({{false}}));
    BOOST_CHECK_THROW(v.write<std::string>({{"test"}}), ChimeraTK::logic_error);

    BOOST_CHECK_NO_THROW(v.read<int8_t>());
    BOOST_CHECK_NO_THROW(v.read<int16_t>());
    BOOST_CHECK_NO_THROW(v.read<int32_t>());
    BOOST_CHECK_NO_THROW(v.read<int64_t>());
    BOOST_CHECK_NO_THROW(v.read<uint8_t>());
    BOOST_CHECK_NO_THROW(v.read<uint16_t>());
    BOOST_CHECK_NO_THROW(v.read<uint32_t>());
    BOOST_CHECK_NO_THROW(v.read<uint64_t>());
    BOOST_CHECK_NO_THROW(v.read<float>());
    BOOST_CHECK_NO_THROW(v.read<double>());
    BOOST_CHECK_NO_THROW(v.read<bool>());
    BOOST_CHECK_THROW(v.read<std::string>(), ChimeraTK::logic_error);
  };
  for(auto& reg : numeric) {
    runTests(reg);
  }
}

BOOST_AUTO_TEST_CASE(testStringTypeConversions) {
  std::vector<TestBackend::Register> strings;

  std::copy_if(registerList.begin(), //
      registerList.end(),            //
      std::back_inserter(strings),
      [](TestBackend::Register& r) { return r.getType() == TestBackend::Register::Type::String; });

  auto runTests = [](TestBackend::Register& r) {
    BOOST_CHECK_THROW(r.write<int8_t>({{2}}), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(r.write<int16_t>({{2}}), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(r.write<int32_t>({{2}}), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(r.write<uint8_t>({{2}}), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(r.write<uint16_t>({{2}}), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(r.write<uint32_t>({{2}}), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(r.write<uint64_t>({{2}}), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(r.write<float>({{2}}), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(r.write<double>({{2}}), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(r.write<bool>({{false}}), ChimeraTK::logic_error);
    BOOST_CHECK_NO_THROW(r.write<std::string>({{"test"}}));

    BOOST_CHECK_THROW(r.read<int8_t>(), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(r.read<int16_t>(), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(r.read<int32_t>(), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(r.read<uint8_t>(), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(r.read<uint16_t>(), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(r.read<uint32_t>(), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(r.read<uint64_t>(), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(r.read<float>(), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(r.read<double>(), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(r.read<bool>(), ChimeraTK::logic_error);
    BOOST_CHECK_NO_THROW(r.read<std::string>());

    auto v = r.getView({r.getShape(), 0, 0});
    BOOST_CHECK_THROW(v.write<int8_t>({{2}}), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(v.write<int16_t>({{2}}), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(v.write<int32_t>({{2}}), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(v.write<uint8_t>({{2}}), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(v.write<uint16_t>({{2}}), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(v.write<uint32_t>({{2}}), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(v.write<uint64_t>({{2}}), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(v.write<float>({{2}}), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(v.write<double>({{2}}), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(v.write<bool>({{false}}), ChimeraTK::logic_error);
    BOOST_CHECK_NO_THROW(v.write<std::string>({{"test"}}));

    BOOST_CHECK_THROW(v.read<int8_t>(), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(v.read<int16_t>(), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(v.read<int32_t>(), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(v.read<uint8_t>(), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(v.read<uint16_t>(), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(v.read<uint32_t>(), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(v.read<uint64_t>(), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(v.read<float>(), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(v.read<double>(), ChimeraTK::logic_error);
    BOOST_CHECK_THROW(v.read<bool>(), ChimeraTK::logic_error);
    BOOST_CHECK_NO_THROW(v.read<std::string>());
  };
  for(auto& reg : strings) {
    runTests(reg);
  }
}
BOOST_AUTO_TEST_CASE(testGetName) {
  unsigned int i = 0;
  for(auto& r : registerList) {
    BOOST_CHECK(names[i] == r.getName());
    i++;
  }
}
BOOST_AUTO_TEST_CASE(testGetAccessMode) {
  unsigned int i = 0;
  for(auto& r : registerList) {
    BOOST_CHECK(access[i] == r.getAccessMode());
    i++;
  }
}
BOOST_AUTO_TEST_CASE(testGetType) {
  unsigned int i = 0;
  for(auto& r : registerList) {
    BOOST_CHECK(types[i] == r.getType());
    i++;
  }
}
BOOST_AUTO_TEST_SUITE_END()

struct TestView {
  std::vector<std::vector<IntegralType>> d{{1, 3, 4}, //
      {9, 4, 2},                                      //
      {7, 6, 3}};
  std::vector<std::vector<int>> e;

  TestBackend::Register r{"testView",    //
      TestBackend::Register::Access::rw, //
      d};

  TestBackend::Register::Shape s{2, 2};
  size_t x_offset{1};
  size_t y_offset{1};

  TestBackend::Register::View v;

  TestView() : e(convertTo<int>(d)), v(r.getView({s, x_offset, y_offset})) {}
};

BOOST_FIXTURE_TEST_SUITE(viewTests, TestView)

BOOST_AUTO_TEST_CASE(testReadView) {
  auto expected = slice(e, {s, x_offset, y_offset});
  auto data = v.read<int>();
  BOOST_CHECK(data == expected);
}
BOOST_AUTO_TEST_CASE(testWriteView) {
  std::vector<std::vector<int>> expected{{7, 9}, //
      {3, 3}};
  v.write(expected);
  auto data = v.read<int>();
  BOOST_CHECK(data == expected);
}
BOOST_AUTO_TEST_CASE(testCreateView) {
  BOOST_CHECK_NO_THROW(r.getView({r.getShape(), 0, 0}));
  BOOST_CHECK_NO_THROW(r.getView({s, x_offset, y_offset}));
  BOOST_CHECK_NO_THROW((TestBackend::Register::View{r, {s, x_offset, y_offset}}));

  BOOST_CHECK_THROW(r.getView({s, x_offset + 100, y_offset}), ChimeraTK::runtime_error);
  BOOST_CHECK_THROW((TestBackend::Register::View{r, {s, x_offset + 100, y_offset}}), ChimeraTK::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()

template<typename To, typename From>
std::vector<std::vector<To>> convertTo(std::vector<std::vector<From>> const& i) {
  std::vector<std::vector<To>> result;
  for(auto const& row : i) {
    result.emplace_back(std::vector<To>{});
    for(auto const& elem : row) {
      result.back().push_back(static_cast<To>(elem));
    }
  }
  return result;
}

template<typename T>
std::vector<std::vector<T>> slice(std::vector<std::vector<T>> const& i, TestBackend::Register::Window const& w) {
  using Container = std::vector<std::vector<T>>;

  auto rowBegin = (w.row_offset);

  auto rowLimit = (rowBegin + w.shape.rowSize()) < i.size() ? (rowBegin + w.shape.rowSize()) : i.size();

  auto columnBegin = w.column_offset;

  auto columnEnd =
      (columnBegin + w.shape.columnSize()) < i[0].size() ? (columnBegin + w.shape.columnSize()) : i[0].size();
  Container result;

  for(auto row = rowBegin; row < rowLimit; row++) {
    result.push_back({});
    for(auto colulmn = columnBegin; colulmn < columnEnd; colulmn++) {
      result.back().push_back(i[row][colulmn]);
    }
  }
  return result;
}
