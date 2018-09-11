#define BOOST_TEST_MODULE testDBaseElem_new
#include <boost/test/included/unit_test.hpp>
#include <boost/variant.hpp>

#include "VariantTypes.h"
#include "new_class.h"

template <typename To, typename From>
std::vector<std::vector<To> > convert(std::vector<std::vector<From> > const& i);

BOOST_AUTO_TEST_CASE(testRegisterRead) {
  auto values = std::vector<std::vector<IntegralType> >{ { 1, 3, 5 } };
  TestBackend::Register e{ "integral",                        //
                           TestBackend::Register::Access::rw, //
                           values };

  auto expected = convert<int>(values);
  auto data = e.read<int>();
  BOOST_CHECK(expected == data);
}

BOOST_AUTO_TEST_CASE(testRegisterWrite) {
  TestBackend::Register r{ "integral",
                           TestBackend::Register::Access::rw,
                           TestBackend::Register::Type::Integer,
                           { 1, 3 } };
  auto expected = std::vector<std::vector<int> >{ { 2, 3, 5 } };
  r.write(expected);
  auto data = r.read<int>();

  BOOST_CHECK(expected == data);
  BOOST_CHECK_THROW(
      r.write(std::vector<std::vector<int> >{ { 1, 2, 3, 4, 5 } }),
      std::runtime_error);
}

BOOST_AUTO_TEST_CASE(testGetShape) {
  auto r = TestBackend::Register{ "",
                                  TestBackend::Register::Access::rw,
                                  TestBackend::Register::Type::Integer,
                                  { 4, 4 } };
  BOOST_CHECK((r.getShape() == TestBackend::Register::Shape{ 4, 4 }));

  // check for invalid shape
  BOOST_CHECK_THROW(
      (TestBackend::Register{ "",
                              TestBackend::Register::Access::rw,
                              TestBackend::Register::Type::Integer,
                              { 0, 1 } }),
      std::logic_error);
  BOOST_CHECK_THROW(
      (TestBackend::Register{ "", TestBackend::Register::Access::rw,
                              std::vector<std::vector<IntegralType> >{ {} } }),
      std::logic_error);
}

BOOST_AUTO_TEST_CASE(testIntegralTypeConversions) {
  auto r = TestBackend::Register{ "", TestBackend::Register::Access::rw,
                                  TestBackend::Register::Type::Integer };

 BOOST_CHECK_NO_THROW(r.write<int8_t>({{-2}}));
 BOOST_CHECK_NO_THROW(r.write<int16_t>({{-2}}));
 BOOST_CHECK_NO_THROW(r.write<int32_t>({{-2}}));
 BOOST_CHECK_NO_THROW(r.write<int64_t>({{-2}}));
 BOOST_CHECK_NO_THROW(r.write<uint8_t>({{2}}));
 BOOST_CHECK_NO_THROW(r.write<uint16_t>({{2}}));
 BOOST_CHECK_NO_THROW(r.write<uint32_t>({{2}}));
 BOOST_CHECK_NO_THROW(r.write<uint64_t>({{2}}));
 BOOST_CHECK_NO_THROW(r.write<float>({{2}}));
 BOOST_CHECK_NO_THROW(r.write<double>({{2}}));
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
}

template <typename To, typename From>
std::vector<std::vector<To> > convert(std::vector<std::vector<From> > const& i) {
  std::vector<std::vector<To> > result;
  for (auto const& row : i) {
    result.emplace_back(std::vector<To>{});
    for (auto const& elem : row) {
      result.back().push_back(static_cast<To>(elem));
    }
  }
  return result;
}
