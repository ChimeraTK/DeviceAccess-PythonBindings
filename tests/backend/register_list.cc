#include "register_list.h"

#include <limits>

namespace TestBackend {

class TBRegisterInfo : public ChimeraTK::RegisterInfo {
  ChimeraTK::RegisterPath name_;
  size_t elements_;
  size_t channels_;
  size_t dimensions_;
  DataDescriptor descriptor_;

public:
  TBRegisterInfo(ChimeraTK::RegisterPath name, size_t elements, size_t channels,
                 size_t dimensions,
                 ChimeraTK::RegisterInfo::DataDescriptor descriptor)
      : name_(std::move(name)),  //
        elements_(elements),     //
        channels_(channels),     //
        dimensions_(dimensions), //
        descriptor_(std::move(descriptor)) {}

  TBRegisterInfo() = default;
  virtual ~TBRegisterInfo() override = default;

  ChimeraTK::RegisterPath getRegisterName() const override { return name_; }

  unsigned int getNumberOfElements() const override {
    return static_cast<unsigned int>(elements_);
  }
  unsigned int getNumberOfChannels() const override {
    return static_cast<unsigned int>(channels_);
  }
  unsigned int getNumberOfDimensions() const override {
    return static_cast<unsigned int>(dimensions_);
  }
  DataDescriptor const &getDataDescriptor() const override {
    return descriptor_;
  }
};
boost::shared_ptr<TBRegisterInfo> getChimeraTkRegisterInfo(Register const &r);
std::tuple<size_t, size_t, size_t>
convertToChimeraTkShape(Register::Shape const &s);
ChimeraTK::RegisterInfo::DataDescriptor
getChimeraTkRegisterDescriptor(Register const &r);

ChimeraTK::RegisterCatalogue convertToRegisterCatalogue(RegisterList const &l) {
  ChimeraTK::RegisterCatalogue catalogue;
  for (auto const &elem : l) {
    catalogue.addRegister(getChimeraTkRegisterInfo(elem.second));
  }
  return catalogue;
}
boost::shared_ptr<TBRegisterInfo> getChimeraTkRegisterInfo(Register const &r) {
  size_t channels;
  size_t elements;
  size_t dimensions;
  std::tie(channels, elements, dimensions) =
      convertToChimeraTkShape(r.getShape());

  return boost::shared_ptr<TBRegisterInfo>(
      new TBRegisterInfo(r.getName(), //
                         elements,    //
                         channels,    //
                         dimensions,  //
                         getChimeraTkRegisterDescriptor(r)));
}
std::tuple<size_t, size_t, size_t>
convertToChimeraTkShape(Register::Shape const &s) {
  size_t dimensions = 0;
  size_t channels = s.rowSize();
  size_t elements = s.columnSize();

  if (channels == 1) {
    if (elements == 1) {
      dimensions = 0; // scalar; as only one element
    } else {
      dimensions = 1;
    }
  } else if (channels > 1) {
    dimensions = 2;
  }
  return std::tuple<size_t, size_t, size_t>{channels, elements, dimensions};
}
ChimeraTK::RegisterInfo::DataDescriptor
getChimeraTkRegisterDescriptor(Register const &r) {
  using RegisterType = TestBackend::Register::Type;

  ChimeraTK::RegisterInfo::FundamentalType type;
  bool isIntegral{};
  bool isSigned{};
  std::size_t nDigits{};
  std::size_t nFractionalDigits{};

  switch (r.getType()) {
  case RegisterType::Integer:
    type = ChimeraTK::RegisterInfo::FundamentalType::numeric;
    isIntegral = true;
    isSigned = true;
    nDigits = 10;
    nFractionalDigits = 0;
    break;
  case RegisterType::FloatingPoint:
    type = ChimeraTK::RegisterInfo::FundamentalType::numeric;
    isIntegral = false;
    isSigned = true;
    nDigits = 309;
    nFractionalDigits = 15;
    break;
  case RegisterType::Bool:
    type = ChimeraTK::RegisterInfo::FundamentalType::boolean;
    isIntegral = true;
    isSigned = false;
    nDigits = 1;
    nFractionalDigits = 0;
    break;
  case RegisterType::String:
    type = ChimeraTK::RegisterInfo::FundamentalType::string;
    isIntegral = false;
    isSigned = false;
    nDigits = 0;
    nFractionalDigits = 0;
    break;
  }
  return ChimeraTK::RegisterInfo::DataDescriptor(type,       //
                                                 isIntegral, //
                                                 isSigned,   //
                                                 nDigits,    //
                                                 nFractionalDigits);
}
RegisterList getDummyList() {
  return RegisterList{
      {"/scalar/Int",
       Register{
           "/scalar/Int",           //
           Register::Access::rw,    //
           Register::Type::Integer, //
           {1, 1}                   //
       }},
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

      {"/scalar/Double",
       Register{
           "/scalar/Double",                //
           Register::Access::rw,          //
           Register::Type::FloatingPoint, //
           {1, 1}                         //
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

      {"/scalar/Bool",
       Register{
           "/scalar/Bool",         //
           Register::Access::rw, //
           Register::Type::Bool, //
           {1, 1}                //
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

      {"/scalar/String",
       Register{
           "/scalar/String",         //
           Register::Access::rw,   //
           Register::Type::String, //
           {1, 1}                  //
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
Register &search(RegisterList &l, std::string const &id) {
  return l.at(id); // throws std::out_of_range if not found
}
} // namespace TestBackend
