#define BOOST_TEST_MODULE testRegisterList

#include "backend.h"
#include "register_list.h"
#include <ChimeraTK/RegisterCatalogue.h>
#include <boost/test/included/unit_test.hpp>

TestBackend::RegisterList getTestList();

struct Fixture_t {

  TestBackend::RegisterList list_;

  Fixture_t() : list_(getTestList()) {}
};

BOOST_FIXTURE_TEST_CASE(testSearch, Fixture_t) {
  auto &r = TestBackend::search(list_, "test2");
  BOOST_CHECK(r.getName() == "test2");
  BOOST_CHECK_THROW(TestBackend::search(list_, "t"), std::out_of_range);
}

BOOST_FIXTURE_TEST_CASE(testCatalogue, Fixture_t) {
  auto catalogue = convertToRegisterCatalogue(list_);

  BOOST_CHECK(catalogue.getNumberOfRegisters() == list_.size());

  for (auto &it : catalogue) {
    if (it.getRegisterName() == "test1") {

      BOOST_CHECK(it.getRegisterName() == "test1");
      BOOST_CHECK(it.getNumberOfChannels() == 1);
      BOOST_CHECK(it.getNumberOfElements() == 1);
      BOOST_CHECK(it.getNumberOfDimensions() == 0);
      BOOST_CHECK(it.getDataDescriptor().fundamentalType() ==
                  ChimeraTK::RegisterInfo::FundamentalType::numeric);
    } else if (it.getRegisterName() == "test2") {
      BOOST_CHECK(it.getRegisterName() == "test2");
      BOOST_CHECK(it.getNumberOfChannels() == 3);
      BOOST_CHECK(it.getNumberOfElements() == 5);
      BOOST_CHECK(it.getNumberOfDimensions() == 2);
      BOOST_CHECK(it.getDataDescriptor().fundamentalType() ==
                  ChimeraTK::RegisterInfo::FundamentalType::string);
    } else {
      BOOST_FAIL("Bad Catalogue");
    }
  }
}

  TestBackend::RegisterList getTestList() {
    TestBackend::RegisterList tmp;

    tmp.emplace(std::make_pair(
        std::string{"test1"},
        TestBackend::Register{"test1",                              //
                              TestBackend::Register::Access::rw,    //
                              TestBackend::Register::Type::Integer, //
                              {1, 1}}));
    tmp.emplace(std::make_pair(
        std::string{"test2"},
        TestBackend::Register{"test2",                             //
                              TestBackend::Register::Access::rw,   //
                              TestBackend::Register::Type::String, //
                              {3, 5}}));
    return tmp;
  }
