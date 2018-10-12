#define BOOST_TEST_MODULE testRegisterList
#include <boost/test/included/unit_test.hpp>

#include <ChimeraTK/RegisterCatalogue.h>
#include "register_list.h"
#include "backend.h"

TestBackend::RegisterList getTestList();

struct Fixture_t {

  TestBackend::RegisterList list_;

  Fixture_t() : list_(getTestList()) {}
};

BOOST_FIXTURE_TEST_CASE(testSearch, Fixture_t) {
  TestBackend::DBaseElem & e = TestBackend::search(list_, "test2");
  BOOST_CHECK(TestBackend::id(e) == "test2");
  BOOST_CHECK_THROW(TestBackend::search(list_, "t"), std::runtime_error);
}

BOOST_FIXTURE_TEST_CASE(testCatalogue, Fixture_t) {
  auto catalogue = convertToRegisterCatalogue(list_);

  BOOST_CHECK(catalogue.getNumberOfRegisters() == list_.size());

  auto it = catalogue.begin();
  BOOST_CHECK(it->getRegisterName() == "test1");
  BOOST_CHECK(it->getNumberOfChannels() == 0);
  BOOST_CHECK(it->getNumberOfElements() == 0);
  BOOST_CHECK(it->getNumberOfDimensions() == 0);
  BOOST_CHECK(it->getDataDescriptor().fundamentalType() ==
              ChimeraTK::RegisterInfo::FundamentalType::numeric);

  ++it;
  BOOST_CHECK(it->getRegisterName() == "test2");
  BOOST_CHECK(it->getNumberOfChannels() == 3);
  BOOST_CHECK(it->getNumberOfElements() == 5);
  BOOST_CHECK(it->getNumberOfDimensions() == 2);
  BOOST_CHECK(it->getDataDescriptor().fundamentalType() ==
              ChimeraTK::RegisterInfo::FundamentalType::string);
}

TestBackend::RegisterList getTestList() {
  TestBackend::RegisterList tmp;
  using Access = TestBackend::AccessMode;
  tmp.emplace_back(TestBackend::DBaseElem{
    "test1", std::vector<std::vector<TestBackend::Int_t> >(), Access::rw
  });
  tmp.emplace_back(
      TestBackend::DBaseElem{ "test2",
                              std::vector<std::vector<TestBackend::String_t> >(
                                  3, std::vector<TestBackend::String_t>(5)),
                              Access::rw });

  return tmp;
}
