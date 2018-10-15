#include <limits>
#include<boost/shared_ptr.hpp>
#include "register_list.h"


namespace TestBackend {



boost::shared_ptr<TBRegisterInfo> getRegInfo(DBaseElem e);
boost::shared_ptr<TBRegisterInfo> getRegInfo(Register e);
ChimeraTK::RegisterInfo::DataDescriptor createDescriptor(DBaseElem const& e);
std::tuple<ChimeraTK::RegisterInfo::FundamentalType, bool, bool, std::size_t,
           std::size_t>
getTypeInfo(DBaseElem const& e);

RegisterList getDummyList() {
  RegisterList l;
  using Access = TestBackend::AccessMode;

  using Inner_i = std::vector<TestBackend::Int_t>;
  using Outer_i = std::vector<Inner_i>;

  using Inner_d = std::vector<TestBackend::Double_t>;
  using Outer_d = std::vector<Inner_d>;

  using Inner_b = std::vector<TestBackend::Bool_t>;
  using Outer_b = std::vector<Inner_b>;

  using Inner_s = std::vector<TestBackend::String_t>;
  using Outer_s = std::vector<Inner_s>;

  /* List of OneD registers */
  l.emplace_back(DBaseElem{ "/oneD/Int", Outer_i(1, Inner_i(5)), Access::rw });
  l.emplace_back(
      DBaseElem{ "/oneD/Double", Outer_d(1, Inner_d(5)), Access::rw });
  l.emplace_back(DBaseElem{ "/oneD/Bool", Outer_b(1, Inner_b(5)), Access::rw });
  l.emplace_back(
      DBaseElem{ "/oneD/String", Outer_s(1, Inner_s(5)), Access::rw });

  /* List o_backf TwoD registers */
  l.emplace_back(DBaseElem{ "/TwoD/Int", Outer_i(4, Inner_i(3)), Access::rw });
  l.emplace_back(
      DBaseElem{ "/TwoD/Double", Outer_d(4, Inner_d(3)), Access::rw });
  l.emplace_back(DBaseElem{ "/TwoD/Bool", Outer_b(4, Inner_b(3)), Access::rw });
  l.emplace_back(
      DBaseElem{ "/TwoD/String", Outer_s(4, Inner_s(3)), Access::rw });

  /* List of readOnly registers with fixed content */
  l.emplace_back(
      DBaseElem{ "/ReadOnly/Int", Outer_i(1, Inner_i{ 100 }), Access::ro });
  l.emplace_back(DBaseElem{ "/ReadOnly/Double",             //
                            Outer_d(1, Inner_d{ 98.878 }), //
                            Access::ro });
  l.emplace_back(
      DBaseElem{ "/ReadOnly/Bool", Outer_b(1, Inner_b{ true }), Access::ro });
  l.emplace_back(DBaseElem{ "/ReadOnly/String",                     //
                            Outer_s(1, Inner_s{ "fixed_string" }), //
                            Access::ro });

  return l;
}

ChimeraTK::RegisterCatalogue convertToRegisterCatalogue(List const &l){
  ChimeraTK::RegisterCatalogue catalogue;
  for (auto const& elem : l) {
    catalogue.addRegister(getRegInfo(elem.second));
  }
  return catalogue;
}
ChimeraTK::RegisterCatalogue convertToRegisterCatalogue(const RegisterList& l) {
  ChimeraTK::RegisterCatalogue catalogue;
  for (auto const& elem : l) {
    catalogue.addRegister(getRegInfo(elem));
  }
  return catalogue;
}



boost::shared_ptr<TBRegisterInfo> getRegInfo(DBaseElem e) {

  return boost::shared_ptr<TBRegisterInfo>(
      new TBRegisterInfo(id(e), e.getElements(), //
                         e.getChannels(),        //
                         e.getDimensions(),      //
                         createDescriptor(e)));
}
ChimeraTK::RegisterInfo::DataDescriptor createDescriptor(DBaseElem const& e) {

  ChimeraTK::RegisterInfo::FundamentalType type;
  bool isIntegral;
  bool isSigned;
  std::size_t nDigits;
  std::size_t nFractionalDigits;
  std::tie(type, isIntegral, isSigned, nDigits, nFractionalDigits) =
      getTypeInfo(e);

  return ChimeraTK::RegisterInfo::DataDescriptor(type, isIntegral, isSigned,
                                              nDigits, nFractionalDigits);
}

std::tuple<ChimeraTK::RegisterInfo::FundamentalType, bool, bool, std::size_t,
           std::size_t>
getTypeInfo(DBaseElem const& e) {

  using Type_t = TestBackend::ElementType;

  ChimeraTK::RegisterInfo::FundamentalType type;
  bool isIntegral;
  bool isSigned;
  std::size_t nDigits;
  std::size_t nFractionalDigits;

  switch (TestBackend::type(e)) {

    case Type_t::Int:
      type = ChimeraTK::RegisterInfo::FundamentalType::numeric;
      isIntegral = true;
      isSigned = true;
      nDigits = 10;
      nFractionalDigits = 0;
      break;

    case Type_t::Double:
      type = ChimeraTK::RegisterInfo::FundamentalType::numeric;
      isIntegral = false;
      isSigned = true;
      nDigits = 309;
      nFractionalDigits = 15;
      break;

    case Type_t::Bool:
      type = ChimeraTK::RegisterInfo::FundamentalType::boolean;
      isIntegral = true;
      isSigned = false;
      nDigits = 1;
      nFractionalDigits = 0;
      break;

    case Type_t::String:
      type = ChimeraTK::RegisterInfo::FundamentalType::string;
      isIntegral = false;
      isSigned = false;
      nDigits = 0;
      nFractionalDigits = 0;
      break;
  }

  return std::make_tuple(type, isIntegral, isSigned, nDigits,
                         nFractionalDigits);
}

DBaseElem& search(RegisterList& l, std::string const& id) {
  for (auto& e : l) {
    if (e == id) {
      return e;
    }
  }
  std::string error_message = "Register " + id + " could not be found";
  throw std::runtime_error(error_message);
}

boost::shared_ptr<TBRegisterInfo> getRegInfo(Register e) {
  auto t = ChimeraTK::RegisterInfo::FundamentalType::string;
  auto a = ChimeraTK::RegisterInfo::DataDescriptor(t, true, false, 5, 2);
  return boost::shared_ptr<TBRegisterInfo>(new TBRegisterInfo("", 6, 3, 1, a));
   //return boost::shared_ptr<TBRegisterInfo>(new TBRegisterInfo());
}
List getList(){
    return List{
        {"/oneD/Int",
         Register{
             "/oneD/Int",             //
             Register::Access::rw,    //
             Register::Type::Integer, //
             {1, 5}                   //
         }},
        {"/twoD/Int",
         Register{
             "/twoD/Int",             //
             Register::Access::rw,    //
             Register::Type::Integer, //
             {4, 3}                   //
         }},

        {"/oneD/Double",
         Register{
             "/oneD/Double",                //
             Register::Access::rw,          //
             Register::Type::FloatingPoint, //
             {1, 5}                         //
         }},
        {"/twoD/Double",
         Register{
             "/twoD/Double",                //
             Register::Access::rw,          //
             Register::Type::FloatingPoint, //
             {4, 3}                         //
         }},

        {"/oneD/Bool",
         Register{
             "/oneD/Bool",         //
             Register::Access::rw, //
             Register::Type::Bool, //
             {1, 5}                //
         }},
        {"/twoD/Bool",
         Register{
             "/twoD/Bool",         //
             Register::Access::rw, //
             Register::Type::Bool, //
             {4, 3}                //
         }},

        {"/oneD/String",
         Register{
             "/oneD/String",         //
             Register::Access::rw,   //
             Register::Type::String, //
             {1, 5}                  //
         }},
        {"/twoD/String",
         Register{
             "/twoD/String",         //
             Register::Access::rw,   //
             Register::Type::String, //
             {4, 3}                  //
         }},

    };
}

} // namespace TestBackend
