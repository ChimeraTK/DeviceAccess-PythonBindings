#define BOOST_TEST_MODULE testDBaseElem_new
#include <boost/test/included/unit_test.hpp>
#include <boost/variant.hpp>

#include "new_class.h"
#include "VariantTypes.h"

BOOST_AUTO_TEST_CASE(test_read) {
  // create the class
  // create expected values
  // compare and decide
  auto expected = std::vector<std::vector<IntegralType> >{ { 1, 3, 5 } };

  TestBackend::Register e{ "integral", TestBackend::Register::Access::rw,
                           expected };

  /*
    auto data = e.read<int>();
    BOOST_CHECK(expected == data);
  */
}
