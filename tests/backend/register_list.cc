#include "register_list.h"
#include <limits>

namespace TestBackend {


class TBRegisterInfo : public mtca4u::RegisterInfo {
  mtca4u::RegisterPath name_;
  std::size_t elements_;
  std::size_t channels_;
  std::size_t dimensions_;
  DataDescriptor descriptor_;

public:
  TBRegisterInfo(mtca4u::RegisterPath name, //
                 std::size_t elements,      //
                 std::size_t channels,      //
                 std::size_t dimensions,    //
                 mtca4u::RegisterInfo::DataDescriptor descriptor)
      : name_(std::move(name)),
        elements_(elements),
        channels_(channels),
        dimensions_(dimensions),
        descriptor_(std::move(descriptor)) {}

  TBRegisterInfo() = default;
  TBRegisterInfo(TBRegisterInfo const& rhs) = default;
  TBRegisterInfo(TBRegisterInfo&& rhs) = default;
  TBRegisterInfo& operator=(TBRegisterInfo const& rhs) = default;
  TBRegisterInfo& operator=(TBRegisterInfo&& rhs) = default;
  virtual ~TBRegisterInfo() = default;

  mtca4u::RegisterPath getRegisterName() const override { return name_; }
  unsigned int getNumberOfElements() const override { return elements_; }
  unsigned int getNumberOfChannels() const override { return channels_; }
  unsigned int getNumberOfDimensions() const override { return dimensions_; }
  DataDescriptor const& getDataDescriptor() const override {
    return descriptor_;
  }
};

boost::shared_ptr<TBRegisterInfo> getRegInfo(DBaseElem e);
mtca4u::RegisterInfo::DataDescriptor createDescriptor(DBaseElem const& e);
std::tuple<mtca4u::RegisterInfo::FundamentalType, bool, bool, std::size_t,
           std::size_t>
getTypeInfo(DBaseElem const& e);

RegisterList getRegisterList() {
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

mtca4u::RegisterCatalogue getRegisterCatalogue(const RegisterList& l) {
  mtca4u::RegisterCatalogue catalogue;
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
mtca4u::RegisterInfo::DataDescriptor createDescriptor(DBaseElem const& e) {

  mtca4u::RegisterInfo::FundamentalType type;
  bool isIntegral;
  bool isSigned;
  std::size_t nDigits;
  std::size_t nFractionalDigits;
  std::tie(type, isIntegral, isSigned, nDigits, nFractionalDigits) =
      getTypeInfo(e);

  return mtca4u::RegisterInfo::DataDescriptor(type, isIntegral, isSigned,
                                              nDigits, nFractionalDigits);
}

std::tuple<mtca4u::RegisterInfo::FundamentalType, bool, bool, std::size_t,
           std::size_t>
getTypeInfo(DBaseElem const& e) {

  using Type_t = TestBackend::ElementType;

  mtca4u::RegisterInfo::FundamentalType type;
  bool isIntegral;
  bool isSigned;
  std::size_t nDigits;
  std::size_t nFractionalDigits;

  switch (TestBackend::type(e)) {

    case Type_t::Int:
      type = mtca4u::RegisterInfo::FundamentalType::numeric;
      isIntegral = true;
      isSigned = true;
      nDigits = 10;
      nFractionalDigits = 0;
      break;

    case Type_t::Double:
      type = mtca4u::RegisterInfo::FundamentalType::numeric;
      isIntegral = false;
      isSigned = true;
      nDigits = 309;
      nFractionalDigits = 15;
      break;

    case Type_t::Bool:
      type = mtca4u::RegisterInfo::FundamentalType::numeric;
      isIntegral = true;
      isSigned = false;
      nDigits = 1;
      nFractionalDigits = 0;
      break;

    case Type_t::String:
      type = mtca4u::RegisterInfo::FundamentalType::string;
      isIntegral = false;
      isSigned = false;
      nDigits = 0;
      nFractionalDigits = 0;
      break;
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


} // namespace TestBackend
