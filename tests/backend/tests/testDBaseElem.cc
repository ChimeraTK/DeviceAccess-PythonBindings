#define BOOST_TEST_MODULE testDBaseElem
#include <boost/test/included/unit_test.hpp>
#include <boost/variant.hpp>

#include "register_access.h"

namespace tb = TestBackend;
using Access = tb::AccessMode;

std::vector<tb::Int_t> iRange(int start, int stop);
std::vector<int> convertToint(std::vector<tb::Int_t> const& from);
std::vector<tb::String_t> StringList();

BOOST_AUTO_TEST_CASE(testConstruction) {
  using Type = tb::ElementType;

  auto e = TestBackend::DBaseElem("test",                                  //
                                  std::vector<std::vector<tb::Bool_t> >(), //
                                  Access::ro);

  BOOST_CHECK(tb::type(e) == Type::Bool);
  BOOST_CHECK(tb::id(e) == "test");
  BOOST_CHECK(tb::access(e) == Access::ro);
  BOOST_CHECK(e.getChannels() == 0);
  BOOST_CHECK(e.getElements() == 0);
}

/*============================================================================*/
struct DBElemFixture {

  tb::DBaseElem oneChannelInt;
  tb::DBaseElem multiChannelInt;
  tb::DBaseElem multiChannelDouble;
  tb::DBaseElem multiChannelString;
  tb::DBaseElem multiChannelBool;
  tb::DBaseElem rwAccess;
  tb::DBaseElem roAccess;
  tb::DBaseElem woAccess;

  DBElemFixture()
      : oneChannelInt("oneChannelInt",
                      std::vector<std::vector<tb::Int_t> >(1, iRange(0, 10))),
        multiChannelInt("multiChannelInt",
                        std::vector<std::vector<tb::Int_t> >(3, iRange(0, 10))),
        multiChannelDouble("multiChannelDouble",
                           std::vector<std::vector<tb::Double_t> >(
                               2, std::vector<tb::Double_t>(3, 10.98))),
        multiChannelString(
            "multiChannelString",
            std::vector<std::vector<tb::String_t> >(2, StringList())),
        multiChannelBool("multiChannelBool",
                         std::vector<std::vector<tb::Bool_t> >(
                             2, std::vector<tb::Bool_t>(10, true))),
        rwAccess("rwAccess", std::vector<std::vector<tb::Int_t> >(
                                 1, std::vector<tb::Int_t>(10, 5))),
        roAccess("roAccess", std::vector<std::vector<tb::Int_t> >(
                                 1, std::vector<tb::Int_t>(10, 5)),
                 Access::ro),
        woAccess("woAccess", std::vector<std::vector<tb::Int_t> >(
                                 1, std::vector<tb::Int_t>(10, 5)),
                 Access::wo) {}
};
/*============================================================================*/

BOOST_FIXTURE_TEST_CASE(testGetContent_OneD, DBElemFixture) {

  std::vector<std::vector<int> > container(1, std::vector<int>(5, int {}));
  auto x_offset = 2;
  auto y_offset = 0;
  tb::copyFrom(oneChannelInt, container, x_offset, y_offset);
  BOOST_CHECK(container.size() == 1);
  BOOST_CHECK(container[0].size() == 5);
  BOOST_CHECK(container[0] == convertToint(iRange(2, 7)));

  auto x_size = 5;
  auto y_size = 0;
  x_offset = 6;
  y_offset = 0;

  auto data =
      tb::copyAs<int>(oneChannelInt, x_size, y_size, x_offset, y_offset);
  BOOST_CHECK(data.size() == 1);
  BOOST_CHECK(data[0].size() == 4);
  BOOST_CHECK(data[0] == convertToint(iRange(6, 10)));

  x_size = 5;
  y_size = 0;
  x_offset = 2;
  y_offset = 0;
  data = tb::copyAs<int>(oneChannelInt, x_size, y_size, x_offset, y_offset);
  BOOST_CHECK(data.size() == 1);
  BOOST_CHECK(data[0].size() == 5);
  BOOST_CHECK(data[0] == convertToint(iRange(2, 7)));
}

BOOST_FIXTURE_TEST_CASE(testGetContent_TwoD, DBElemFixture) {

  std::vector<std::vector<int> > container(3, std::vector<int>(5, int {}));
  auto x_offset = 2;
  auto y_offset = 0;
  tb::copyFrom(multiChannelInt, container, x_offset, y_offset);
  BOOST_CHECK(container.size() == 3);
  BOOST_CHECK(container[1].size() == 5);
  BOOST_CHECK(container[2] == convertToint(iRange(2, 7)));

  auto x_size = 5;
  auto y_size = 0;
  x_offset = 6;
  y_offset = 0;
  auto data =
      tb::copyAs<int>(multiChannelInt, x_size, y_size, x_offset, y_offset);
  BOOST_CHECK(data.size() == 3);
  BOOST_CHECK(data[1].size() == 4);
  BOOST_CHECK(data[0] == convertToint(iRange(6, 10)));

  x_size = 5;
  y_size = 0;
  x_offset = 2;
  y_offset = 0;
  data = tb::copyAs<int>(multiChannelInt, x_size, y_size, x_offset, y_offset);
  BOOST_CHECK(data.size() == 3);
  BOOST_CHECK(data[2].size() == 5);
  BOOST_CHECK(data[1] == convertToint(iRange(2, 7)));
}

BOOST_FIXTURE_TEST_CASE(testCopyAs_saftey, DBElemFixture) {

  auto data = tb::copyAs<int>(multiChannelInt);
  auto expected =
      std::vector<std::vector<int> >(3, convertToint(iRange(0, 10)));
  BOOST_CHECK(data == expected);

  // smaller chunk
  auto x_size = 5;
  auto y_size = 1;
  auto x_offset = 1;
  auto y_offset = 1;
  data = tb::copyAs<int>(multiChannelInt, x_size, y_size, x_offset, y_offset);
  expected = std::vector<std::vector<int> >(
      1, convertToint(iRange(x_offset, x_offset + x_size)));
  BOOST_CHECK(data == expected);

  // request exceeds underlying size
  x_size = 15;
  y_size = 5;
  data = tb::copyAs<int>(multiChannelInt, x_size, y_size);
  expected = std::vector<std::vector<int> >(3, convertToint(iRange(0, 10)));
  BOOST_CHECK(data == expected);
}

BOOST_FIXTURE_TEST_CASE(testValidateContainer, DBElemFixture) {
  BOOST_CHECK(tb::isSubShape(multiChannelInt, 10, 3, 0, 0) == true);
  BOOST_CHECK(tb::isSubShape(multiChannelInt, 10, 3, 1, 0) == false);
  BOOST_CHECK(tb::isSubShape(multiChannelInt, 10, 3, 0, 1) == false);

  BOOST_CHECK(tb::isSubShape(multiChannelInt, 5, 2, 2, 1) == true);
  BOOST_CHECK(tb::isSubShape(multiChannelInt, 11, 4, 0, 0) == false);
}

BOOST_FIXTURE_TEST_CASE(testCopyFrom_saftey, DBElemFixture) {
  auto container = std::vector<std::vector<int> >(3, std::vector<int>(10, 0));
  auto expected =
      std::vector<std::vector<int> >(3, convertToint(iRange(0, 10)));
  std::size_t copied_x_size;
  std::size_t copied_y_size;
  std::tie(copied_x_size, copied_y_size) = tb::copyFrom(multiChannelInt, //
                                                        container);
  BOOST_CHECK(copied_x_size == 10);
  BOOST_CHECK(copied_y_size == 3);
  BOOST_CHECK(container == expected);

  // offset
  auto x_offset = 2;
  auto y_offset = 1;
  container = std::vector<std::vector<int> >(2, std::vector<int>(5, 0));
  expected = std::vector<std::vector<int> >(
      2, //
      convertToint(iRange(x_offset, x_offset + 5)));
  std::tie(copied_x_size, copied_y_size) = tb::copyFrom(multiChannelInt,     //
                                                        container, x_offset, //
                                                        y_offset);
  BOOST_CHECK(copied_x_size == 5);
  BOOST_CHECK(copied_y_size == 2);
  BOOST_CHECK(container == expected);

  container = std::vector<std::vector<int> >(3, std::vector<int>(10, 0));
  expected = std::vector<std::vector<int> >(3, convertToint(iRange(0, 10)));
  std::tie(copied_x_size, copied_y_size) = tb::copyFrom(multiChannelInt, //
                                                        container);
  BOOST_CHECK(copied_x_size == 10);
  BOOST_CHECK(copied_y_size == 3);
  BOOST_CHECK(container == expected);
}

BOOST_FIXTURE_TEST_CASE(testCopyinto, DBElemFixture) {
  auto data = std::vector<std::vector<int> >(3, std::vector<int>(10, 50));
  std::size_t written_x_size;
  std::size_t written_y_size;
  std::tie(written_x_size, written_y_size) = tb::copyInto(multiChannelInt, //
                                                          data);
  auto readData = tb::copyAs<int>(multiChannelInt);
  BOOST_CHECK(written_x_size = 10);
  BOOST_CHECK(written_y_size = 3);
  BOOST_CHECK(data == readData);

  data = std::vector<std::vector<int> >(5, std::vector<int>(15, 20));
  auto expected = std::vector<std::vector<int> >(3, std::vector<int>(10, 20));
  std::tie(written_x_size, written_y_size) = tb::copyInto(multiChannelInt, //
                                                          data);
  readData = tb::copyAs<int>(multiChannelInt);
  BOOST_CHECK(written_x_size = 10);
  BOOST_CHECK(written_y_size = 3);
  BOOST_CHECK(expected == readData);

  auto x_size = 5;
  auto y_size = 2;
  data = std::vector<std::vector<int> >(y_size, std::vector<int>(x_size, 15));
  auto x_offset = 2;
  auto y_offset = 1;

  std::tie(written_x_size, written_y_size) = tb::copyInto(multiChannelInt, //
                                                          data,            //
                                                          x_offset,        //
                                                          y_offset);
  readData =
      tb::copyAs<int>(multiChannelInt, x_size, y_size, x_offset, y_offset);
  BOOST_CHECK(written_x_size = x_size);
  BOOST_CHECK(written_y_size = y_size);
  BOOST_CHECK(data == readData);

}

BOOST_FIXTURE_TEST_CASE(testConversions_fromString, DBElemFixture) {
  BOOST_CHECK_THROW(tb::copyAs<int>(multiChannelString), std::runtime_error);
  BOOST_CHECK_THROW(tb::copyAs<double>(multiChannelString), std::runtime_error);
  BOOST_CHECK_THROW(tb::copyAs<bool>(multiChannelString), std::runtime_error);

  BOOST_CHECK_THROW(tb::copyAs<int8_t>(multiChannelString), std::runtime_error);
  ;
  BOOST_CHECK_THROW(tb::copyAs<uint8_t>(multiChannelString),
                    std::runtime_error);
  BOOST_CHECK_THROW(tb::copyAs<int16_t>(multiChannelString),
                    std::runtime_error);
  BOOST_CHECK_THROW(tb::copyAs<uint16_t>(multiChannelString),
                    std::runtime_error);
  BOOST_CHECK_THROW(tb::copyAs<int32_t>(multiChannelString),
                    std::runtime_error);
  BOOST_CHECK_THROW(tb::copyAs<uint32_t>(multiChannelString),
                    std::runtime_error);
  BOOST_CHECK_THROW(tb::copyAs<int64_t>(multiChannelString),
                    std::runtime_error);
  BOOST_CHECK_THROW(tb::copyAs<uint64_t>(multiChannelString),
                    std::runtime_error);
  BOOST_CHECK_THROW(tb::copyAs<float>(multiChannelString), std::runtime_error);

  BOOST_CHECK_NO_THROW(tb::copyAs<std::string>(multiChannelString));
}

BOOST_FIXTURE_TEST_CASE(testConversions_fromInt, DBElemFixture) {
  BOOST_CHECK_NO_THROW(tb::copyAs<int>(multiChannelInt));
  BOOST_CHECK_NO_THROW(tb::copyAs<double>(multiChannelInt));
  BOOST_CHECK_NO_THROW(tb::copyAs<bool>(multiChannelInt));
  BOOST_CHECK_NO_THROW(tb::copyAs<int8_t>(multiChannelInt));
  BOOST_CHECK_NO_THROW(tb::copyAs<uint8_t>(multiChannelInt));
  BOOST_CHECK_NO_THROW(tb::copyAs<int16_t>(multiChannelInt));
  BOOST_CHECK_NO_THROW(tb::copyAs<uint16_t>(multiChannelInt));
  BOOST_CHECK_NO_THROW(tb::copyAs<int32_t>(multiChannelInt));
  BOOST_CHECK_NO_THROW(tb::copyAs<uint32_t>(multiChannelInt));
  BOOST_CHECK_NO_THROW(tb::copyAs<int64_t>(multiChannelInt));
  BOOST_CHECK_NO_THROW(tb::copyAs<uint64_t>(multiChannelInt));
  BOOST_CHECK_NO_THROW(tb::copyAs<float>(multiChannelInt));
  BOOST_CHECK_NO_THROW(tb::copyAs<std::string>(multiChannelInt));
}

BOOST_FIXTURE_TEST_CASE(testConversions_fromDouble, DBElemFixture) {
  BOOST_CHECK_NO_THROW(tb::copyAs<int>(multiChannelDouble));
  BOOST_CHECK_NO_THROW(tb::copyAs<double>(multiChannelDouble));
  BOOST_CHECK_NO_THROW(tb::copyAs<bool>(multiChannelDouble));
  BOOST_CHECK_NO_THROW(tb::copyAs<std::string>(multiChannelDouble));
  BOOST_CHECK_NO_THROW(tb::copyAs<int8_t>(multiChannelDouble));
  BOOST_CHECK_NO_THROW(tb::copyAs<uint8_t>(multiChannelDouble));
  BOOST_CHECK_NO_THROW(tb::copyAs<int16_t>(multiChannelDouble));
  BOOST_CHECK_NO_THROW(tb::copyAs<uint16_t>(multiChannelDouble));
  BOOST_CHECK_NO_THROW(tb::copyAs<int32_t>(multiChannelDouble));
  BOOST_CHECK_NO_THROW(tb::copyAs<uint32_t>(multiChannelDouble));
  BOOST_CHECK_NO_THROW(tb::copyAs<int64_t>(multiChannelDouble));
  BOOST_CHECK_NO_THROW(tb::copyAs<uint64_t>(multiChannelDouble));
  BOOST_CHECK_NO_THROW(tb::copyAs<float>(multiChannelDouble));
  BOOST_CHECK_NO_THROW(tb::copyAs<std::string>(multiChannelDouble));
}

BOOST_FIXTURE_TEST_CASE(testConversions_fromBool, DBElemFixture) {
  BOOST_CHECK_NO_THROW(tb::copyAs<int>(multiChannelBool));
  BOOST_CHECK_NO_THROW(tb::copyAs<double>(multiChannelBool));
  BOOST_CHECK_NO_THROW(tb::copyAs<bool>(multiChannelBool));
  BOOST_CHECK_NO_THROW(tb::copyAs<std::string>(multiChannelBool));
  BOOST_CHECK_NO_THROW(tb::copyAs<int8_t>(multiChannelBool));
  BOOST_CHECK_NO_THROW(tb::copyAs<uint8_t>(multiChannelBool));
  BOOST_CHECK_NO_THROW(tb::copyAs<int16_t>(multiChannelBool));
  BOOST_CHECK_NO_THROW(tb::copyAs<uint16_t>(multiChannelBool));
  BOOST_CHECK_NO_THROW(tb::copyAs<int32_t>(multiChannelBool));
  BOOST_CHECK_NO_THROW(tb::copyAs<uint32_t>(multiChannelBool));
  BOOST_CHECK_NO_THROW(tb::copyAs<int64_t>(multiChannelBool));
  BOOST_CHECK_NO_THROW(tb::copyAs<uint64_t>(multiChannelBool));
  BOOST_CHECK_NO_THROW(tb::copyAs<float>(multiChannelBool));
  BOOST_CHECK_NO_THROW(tb::copyAs<std::string>(multiChannelBool));
}

BOOST_FIXTURE_TEST_CASE(testConversions_copyInto_from_int8_t, DBElemFixture) {
  auto data = std::vector<std::vector<int8_t> >(1, std::vector<int8_t>(1, 0));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelInt, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelBool, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelDouble, data));
  BOOST_CHECK_THROW(tb::copyInto(multiChannelString, data), std::runtime_error);
}

BOOST_FIXTURE_TEST_CASE(testConversions_copyInto_from_uint8_t, DBElemFixture) {
  auto data = std::vector<std::vector<uint8_t> >(1, std::vector<uint8_t>(1, 0));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelInt, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelBool, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelDouble, data));
  BOOST_CHECK_THROW(tb::copyInto(multiChannelString, data), std::runtime_error);
}

BOOST_FIXTURE_TEST_CASE(testConversions_copyInto_from_int16_t, DBElemFixture) {
  auto data = std::vector<std::vector<int16_t> >(1, std::vector<int16_t>(1, 0));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelInt, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelBool, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelDouble, data));
  BOOST_CHECK_THROW(tb::copyInto(multiChannelString, data), std::runtime_error);
}

BOOST_FIXTURE_TEST_CASE(testConversions_copyInto_from_uint16_t, DBElemFixture) {
  auto data =
      std::vector<std::vector<uint16_t> >(1, std::vector<uint16_t>(1, 0));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelInt, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelBool, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelDouble, data));
  BOOST_CHECK_THROW(tb::copyInto(multiChannelString, data), std::runtime_error);
}

BOOST_FIXTURE_TEST_CASE(testConversions_copyInto_from_int32_t, DBElemFixture) {
  auto data = std::vector<std::vector<int32_t> >(1, std::vector<int32_t>(1, 0));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelInt, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelBool, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelDouble, data));
  BOOST_CHECK_THROW(tb::copyInto(multiChannelString, data), std::runtime_error);
}

BOOST_FIXTURE_TEST_CASE(testConversions_copyInto_from_uint32_t, DBElemFixture) {
  auto data =
      std::vector<std::vector<uint32_t> >(1, std::vector<uint32_t>(1, 0));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelInt, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelBool, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelDouble, data));
  BOOST_CHECK_THROW(tb::copyInto(multiChannelString, data), std::runtime_error);
}

BOOST_FIXTURE_TEST_CASE(testConversions_copyInto_from_int64_t, DBElemFixture) {
  auto data = std::vector<std::vector<int64_t> >(1, std::vector<int64_t>(1, 0));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelInt, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelBool, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelDouble, data));
  BOOST_CHECK_THROW(tb::copyInto(multiChannelString, data), std::runtime_error);
}

BOOST_FIXTURE_TEST_CASE(testConversions_copyInto_from_uint64_t, DBElemFixture) {
  auto data =
      std::vector<std::vector<uint64_t> >(1, std::vector<uint64_t>(1, 0));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelInt, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelBool, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelDouble, data));
  BOOST_CHECK_THROW(tb::copyInto(multiChannelString, data), std::runtime_error);
}

BOOST_FIXTURE_TEST_CASE(testConversions_copyInto_from_float, DBElemFixture) {
  auto data = std::vector<std::vector<float> >(1, std::vector<float>(1, 0));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelInt, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelBool, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelDouble, data));
  BOOST_CHECK_THROW(tb::copyInto(multiChannelString, data), std::runtime_error);
}

BOOST_FIXTURE_TEST_CASE(testConversions_copyInto_from_double, DBElemFixture) {
  auto data = std::vector<std::vector<double> >(1, std::vector<double>(1, 0));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelInt, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelBool, data));
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelDouble, data));
  BOOST_CHECK_THROW(tb::copyInto(multiChannelString, data), std::runtime_error);
}

BOOST_FIXTURE_TEST_CASE(testConversions_copyInto_from_string, DBElemFixture) {
  auto data = std::vector<std::vector<std::string> >(1, std::vector<std::string>(1, ""));
  BOOST_CHECK_THROW(tb::copyInto(multiChannelInt, data), std::runtime_error);
  BOOST_CHECK_THROW(tb::copyInto(multiChannelBool, data), std::runtime_error);
  BOOST_CHECK_THROW(tb::copyInto(multiChannelDouble, data), std::runtime_error);
  BOOST_CHECK_NO_THROW(tb::copyInto(multiChannelString, data));
}


BOOST_FIXTURE_TEST_CASE(testSetContent_OneD, DBElemFixture) {

  /*  std::vector<std::vector<int>> container(1, convertToint(iRange(0, 5)));
    auto sequencesCopied = tb::copyInto(oneChannelInt, container, 5);
    auto readBack = tb::copyAs<int>(oneChannelInt, 0, 5 );
    BOOST_CHECK(sequencesCopied = 5);
    BOOST_CHECK(readBack[0].size() == 5);
    BOOST_CHECK(container == readBack);*/
}

BOOST_FIXTURE_TEST_CASE(testCheckAccess, DBElemFixture) {
  auto data = std::vector<std::vector<double> >(1, std::vector<double>(1, 0));
  BOOST_CHECK_NO_THROW(tb::copyAs<float>(rwAccess));
  BOOST_CHECK_NO_THROW(tb::copyInto(rwAccess, data));
  BOOST_CHECK_THROW(tb::copyAs<float>(woAccess), std::runtime_error);
  BOOST_CHECK_THROW(tb::copyInto(roAccess, data), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(testType) {}

BOOST_AUTO_TEST_CASE(testId) {}

std::vector<int> convertToint(std::vector<tb::Int_t> const& from) {
  std::vector<int> tmp;
  for (auto val : from) {
    tmp.push_back(static_cast<int>(val));
  }
  return tmp;
}

std::vector<tb::Int_t> iRange(int start, int stop) {
  auto ret = std::vector<tb::Int_t>();
  for (; start < stop; start++) {
    ret.push_back(start);
  }
  return ret;
}

std::vector<tb::String_t> StringList() {
  auto tmp = std::vector<tb::String_t>{ "a", "b", "c", "d" };
  return tmp;
}

/*

object_t(const tb::Int_t& x) : self_(make_unique<int_model_t>(x))
{ cout << "ctor" << endl; }
object_t(const object_t& x) : self_(make_unique<int_model_t>(*x.self_))
{ cout << "copy" << endl; }
object_t(object_t&&) noexcept = default;
object_t& operator=(const object_t& x)
{ object_t
 return
 *this
 tmp(x);
 = object_t(x);
 *this = move(tmp);
 }
 return *this; }
object_t& operator=(object_t&&) noexcept = default;
friend void draw(const object_t& x, ostream& out, size_t position)
{ x.self_->draw_(out, position); }

===============================================================================
Take away
 Returning objects from functions, passing read-only arguments, and passing
 rvalues as sink arguments do not require copying
 Understanding this can greatly improve the efficiency of your application

 */
