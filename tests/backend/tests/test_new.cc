#define BOOST_TEST_MODULE testDBaseElem_new
#include <boost/test/included/unit_test.hpp>
#include <boost/variant.hpp>

#include "VariantTypes.h"
#include "new_class.h"

// example usage:
// convertTo<int>(floatArray)
template <typename To, typename From>
std::vector<std::vector<To>> convertTo(std::vector<std::vector<From>> const &i);

BOOST_AUTO_TEST_CASE(testRegisterCreation) {
  // check for invalid shape
  BOOST_CHECK_THROW((TestBackend::Register{"",
                                           TestBackend::Register::Access::rw,
                                           TestBackend::Register::Type::Integer,
                                           {1, 0}}),
                    std::logic_error);
  BOOST_CHECK_THROW((TestBackend::Register{"",
                                           TestBackend::Register::Access::rw,
                                           TestBackend::Register::Type::Integer,
                                           {0, 1}}),
                    std::logic_error);
  BOOST_CHECK_THROW((TestBackend::Register{"",
                                           TestBackend::Register::Access::rw,
                                           TestBackend::Register::Type::Integer,
                                           {0, 0}}),
                    std::logic_error);
  BOOST_CHECK_NO_THROW(
      (TestBackend::Register{"",
                             TestBackend::Register::Access::rw,
                             TestBackend::Register::Type::Integer,
                             {1, 1}}));
  BOOST_CHECK_NO_THROW(
      (TestBackend::Register{"", TestBackend::Register::Access::rw,
                             std::vector<std::vector<IntegralType>>{{}}}

       ));
  BOOST_CHECK_NO_THROW(
      (TestBackend::Register{"", TestBackend::Register::Access::rw,
                             std::vector<std::vector<FloatingPointType>>{{}}}

       ));
  BOOST_CHECK_NO_THROW(
      (TestBackend::Register{"", TestBackend::Register::Access::rw,
                             std::vector<std::vector<BooleanType>>{{}}}

       ));
  BOOST_CHECK_NO_THROW(
      (TestBackend::Register{"", TestBackend::Register::Access::rw,
                             std::vector<std::vector<StringType>>{{}}}

       ));
}

BOOST_AUTO_TEST_CASE(testRegisterRead) {
  auto values = std::vector<std::vector<IntegralType>>{{1, 3, 5}};
  TestBackend::Register e{"integral",                        //
                          TestBackend::Register::Access::rw, //
                          values};

  auto expected = convertTo<int>(values);
  auto data = e.read<int>();
  BOOST_CHECK(expected == data);
}

BOOST_AUTO_TEST_CASE(testRegisterWrite) {
  TestBackend::Register r{"integral",
                          TestBackend::Register::Access::rw,
                          TestBackend::Register::Type::Integer,
                          {1, 3}};
  auto expected = std::vector<std::vector<int>>{{2, 3, 5}};
  r.write(expected);
  auto data = r.read<int>();

  BOOST_CHECK(expected == data);
  BOOST_CHECK_THROW(r.write(std::vector<std::vector<int>>{{1, 2, 3, 4, 5}}),
                    std::runtime_error);
}

BOOST_AUTO_TEST_CASE(testGetShape) {
  auto r = TestBackend::Register{"",
                                 TestBackend::Register::Access::rw,
                                 TestBackend::Register::Type::Integer,
                                 {4, 4}};
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
    for (unsigned int i = 0; i < SIZE; i++) {
      registerList.emplace_back(names[i], access[i], types[i]);
    }
  }
};

BOOST_FIXTURE_TEST_SUITE(fixtureTests, FixtureContent)
BOOST_AUTO_TEST_CASE(testNumericTypeConversions) {
  std::vector<TestBackend::Register> numeric;

  std::copy_if(registerList.begin(), //
               registerList.end(),   //
               std::back_inserter(numeric),
               [](TestBackend::Register &r) {
                 return r.getType() != TestBackend::Register::Type::String;
               });

  auto runTests = [](TestBackend::Register &r) {
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
    BOOST_CHECK_THROW(r.write<std::string>({{"test"}}), std::logic_error);

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
    BOOST_CHECK_THROW(r.read<std::string>(), std::logic_error);
  };
  for (auto &reg : numeric) {
    runTests(reg);
  }
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_CASE(testStringTypeConversions) {
  auto r = TestBackend::Register{
      "",                                 //
      TestBackend::Register::Access::rw,  //
      TestBackend::Register::Type::String //
  };

  BOOST_CHECK_THROW(r.write<int8_t>({{2}}), std::logic_error);
  BOOST_CHECK_THROW(r.write<int16_t>({{2}}), std::logic_error);
  BOOST_CHECK_THROW(r.write<int32_t>({{2}}), std::logic_error);
  BOOST_CHECK_THROW(r.write<uint8_t>({{2}}), std::logic_error);
  BOOST_CHECK_THROW(r.write<uint16_t>({{2}}), std::logic_error);
  BOOST_CHECK_THROW(r.write<uint32_t>({{2}}), std::logic_error);
  BOOST_CHECK_THROW(r.write<uint64_t>({{2}}), std::logic_error);
  BOOST_CHECK_THROW(r.write<float>({{2}}), std::logic_error);
  BOOST_CHECK_THROW(r.write<double>({{2}}), std::logic_error);
  BOOST_CHECK_THROW(r.write<bool>({{false}}), std::logic_error);
  BOOST_CHECK_NO_THROW(r.write<std::string>({{"test"}}));

  BOOST_CHECK_THROW(r.read<int8_t>(), std::logic_error);
  BOOST_CHECK_THROW(r.read<int16_t>(), std::logic_error);
  BOOST_CHECK_THROW(r.read<int32_t>(), std::logic_error);
  BOOST_CHECK_THROW(r.read<uint8_t>(), std::logic_error);
  BOOST_CHECK_THROW(r.read<uint16_t>(), std::logic_error);
  BOOST_CHECK_THROW(r.read<uint32_t>(), std::logic_error);
  BOOST_CHECK_THROW(r.read<uint64_t>(), std::logic_error);
  BOOST_CHECK_THROW(r.read<float>(), std::logic_error);
  BOOST_CHECK_THROW(r.read<double>(), std::logic_error);
  BOOST_CHECK_THROW(r.read<bool>(), std::logic_error);
  BOOST_CHECK_NO_THROW(r.read<std::string>());
}

BOOST_AUTO_TEST_CASE(testGetName) {
  std::string regName = "testRegister";
  auto r = TestBackend::Register{
      regName,                            //
      TestBackend::Register::Access::rw,  //
      TestBackend::Register::Type::String //
  };
  BOOST_CHECK(regName == r.getName());
}

BOOST_AUTO_TEST_CASE(testGetAccessMode) {
  auto access = TestBackend::Register::Access::ro;
  auto r = TestBackend::Register{
      "",                                 //
      access,                             //
      TestBackend::Register::Type::String //
  };
  BOOST_CHECK(access == r.getAccessMode());
}

BOOST_AUTO_TEST_CASE(testGetType) {}

template <typename To, typename From>
std::vector<std::vector<To>>
convertTo(std::vector<std::vector<From>> const &i) {
  std::vector<std::vector<To>> result;
  for (auto const &row : i) {
    result.emplace_back(std::vector<To>{});
    for (auto const &elem : row) {
      result.back().push_back(static_cast<To>(elem));
    }
  }
  return result;
}
