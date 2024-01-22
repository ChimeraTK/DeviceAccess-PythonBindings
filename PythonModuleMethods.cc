#include "PythonModuleMethods.h"

#include <ChimeraTK/SupportedUserTypes.h>

#include <limits>
#include <stdexcept>

ChimeraTK::AccessModeFlags mtca4upy::DeviceAccess::convertFlagsFromPython(boost::python::list flaglist) {
  ChimeraTK::AccessModeFlags flags{};
  size_t count = len((flaglist));
  for(size_t i = 0; i < count; i++) {
    flags.add(p::extract<ChimeraTK::AccessMode>(flaglist.pop()));
  }
  return flags;
}

ChimeraTK::VoidRegisterAccessor mtca4upy::DeviceAccess::getVoidRegisterAccessor(
    const ChimeraTK::Device& self, const std::string& registerPath, boost::python::list flaglist) {
  return self.getVoidRegisterAccessor(registerPath, convertFlagsFromPython(flaglist));
}

void mtca4upy::DeviceAccess::activateAsyncRead(ChimeraTK::Device& self) {
  self.activateAsyncRead();
}

ChimeraTK::RegisterCatalogue mtca4upy::DeviceAccess::getRegisterCatalogue(ChimeraTK::Device& self) {
  return self.getRegisterCatalogue();
}

namespace mtca4upy::VoidRegisterAccessor {

  bool write(ChimeraTK::VoidRegisterAccessor& self) {
    return self.write();
  }

  bool writeDestructively(ChimeraTK::VoidRegisterAccessor& self) {
    return self.writeDestructively();
  }

  void read(ChimeraTK::VoidRegisterAccessor& self) {
    return self.read();
  }

  bool readNonBlocking(ChimeraTK::VoidRegisterAccessor& self) {
    return self.readNonBlocking();
  }

  bool readLatest(ChimeraTK::VoidRegisterAccessor& self) {
    return self.readLatest();
  }

} // namespace mtca4upy::VoidRegisterAccessor

np::ndarray mtca4upy::DeviceAccess::read(const ChimeraTK::Device& self, const std::string& registerPath,
    size_t numberOfElements, size_t elementsOffset, boost::python::list flaglist) {
  auto reg = self.getRegisterCatalogue().getRegister(registerPath);
  auto usertype = reg.getDataDescriptor().minimumDataType();

  std::unique_ptr<np::ndarray> arr;

  ChimeraTK::callForTypeNoVoid(usertype, [&](auto arg) {
    using UserType = decltype(arg);
    auto acc = self.getTwoDRegisterAccessor<UserType>(
        registerPath, numberOfElements, elementsOffset, convertFlagsFromPython(flaglist));
    acc.read();
    arr = std::make_unique<np::ndarray>(GeneralRegisterAccessor::copyUserBufferToNpArray(
        acc, convert_usertype_to_dtype(usertype), reg.getNumberOfDimensions()));
  });

  return *arr;
}

void mtca4upy::DeviceAccess::write(const ChimeraTK::Device& self, np::ndarray& arr, const std::string& registerPath,
    size_t numberOfElements, size_t elementsOffset, boost::python::list flaglist) {
  auto usertype = convert_dytpe_to_usertype(arr.get_dtype());

  auto bufferTransfer = [&](auto arg) {
    auto acc = self.getTwoDRegisterAccessor<decltype(arg)>(
        registerPath, numberOfElements, elementsOffset, convertFlagsFromPython(flaglist));
    GeneralRegisterAccessor::copyNpArrayToUserBuffer(acc, arr);
    acc.write();
  };

  ChimeraTK::callForTypeNoVoid(usertype, bufferTransfer);
}

ChimeraTK::DataType mtca4upy::DeviceAccess::convert_dytpe_to_usertype(np::dtype dtype) {
  /*
  Inlcuded UserTypes in ChimeratTK::DataType::TheType:

  none,    ///< The data type/concept does not exist, e.g. there is no raw transfer (do not confuse with Void)
  int8,    ///< Signed 8 bit integer
  uint8,   ///< Unsigned 8 bit integer
  int16,   ///< Signed 16 bit integer
  uint16,  ///< Unsigned 16 bit integer
  int32,   ///< Signed 32 bit integer
  uint32,  ///< Unsigned 32 bit integer
  int64,   ///< Signed 64 bit integer
  uint64,  ///< Unsigned 64 bit integer
  float32, ///< Single precision float
  float64, ///< Double precision float
  string,  ///< std::string
  Boolean, ///< Boolean
  Void     ///< Void

  Possible NumpyTypes:

  numpy.bool_ ///< bool ///< Boolean (True or False) stored as a byte
  numpy.byte  ///< signed char   ///< Platform-defined
  numpy.ubyte ///< unsigned char    ///< Platform-defined
  numpy.short ///< short    ///< Platform-defined
  numpy.ushort    ///< unsigned short  ///< Platform-defined
  numpy.intc  ///< int   ///< Platform-defined
  numpy.uintc ///< unsigned int ///< Platform-defined
  numpy.int_  ///< long  ///< Platform-defined
  numpy.uint  ///< unsigned long ///< Platform-defined
  numpy.longlong  ///< long long ///< Platform-defined
  numpy.ulonglong ///< unsigned long long   ///< Platform-defined
  numpy.half / numpy.float16    ///< Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
  numpy.single    ///< float   ///< Platform-defined single precision float: typically sign bit, 8 bits exponent, 23
  bits mantissa numpy.double    ///< double  ///< Platform-defined double precision float: typically sign bit, 11 bits
  exponent, 52 bits mantissa. numpy.longdouble    ///< long double ///< Platform-defined extended-precision float
  numpy.csingle
  ///< float complex  ///< Complex number, represented by two single-precision floats (real and imaginary components)
  numpy.cdouble
  ///< double complex ///< Complex number, represented by two double-precision floats (real and imaginary components).
  numpy.clongdouble   ///< long double complex    ///< Complex number, represented by two extended-precision floats
  (real and imaginary components).

  */
  auto usertype = ChimeraTK::DataType::none;
  if(dtype == np::dtype::get_builtin<int8_t>())
    usertype = ChimeraTK::DataType::int8;
  else if(dtype == np::dtype::get_builtin<int16_t>())
    usertype = ChimeraTK::DataType::int16;
  else if(dtype == np::dtype::get_builtin<int32_t>())
    usertype = ChimeraTK::DataType::int32;
  else if(dtype == np::dtype::get_builtin<int64_t>())
    usertype = ChimeraTK::DataType::int64;
  else if(dtype == np::dtype::get_builtin<uint8_t>())
    usertype = ChimeraTK::DataType::uint8;
  else if(dtype == np::dtype::get_builtin<uint16_t>())
    usertype = ChimeraTK::DataType::uint16;
  else if(dtype == np::dtype::get_builtin<uint32_t>())
    usertype = ChimeraTK::DataType::uint32;
  else if(dtype == np::dtype::get_builtin<uint64_t>())
    usertype = ChimeraTK::DataType::uint64;
  else if(dtype == np::dtype::get_builtin<float>())
    usertype = ChimeraTK::DataType::float32;
  else if(dtype == np::dtype::get_builtin<double>())
    usertype = ChimeraTK::DataType::float64;
  else if(dtype == np::dtype::get_builtin<bool>())
    usertype = ChimeraTK::DataType::Boolean;
  else
    throw std::invalid_argument("Invalid numpy dtype");
  return usertype;
}

np::dtype mtca4upy::DeviceAccess::convert_usertype_to_dtype(ChimeraTK::DataType usertype) {
  std::unique_ptr<np::dtype> rv;
  ChimeraTK::callForTypeNoVoid(usertype, [&](auto arg) {
    using UserType = decltype(arg);
    if constexpr(std::is_same<UserType, ChimeraTK::Boolean>::value) {
      rv = std::make_unique<np::dtype>(np::dtype::get_builtin<bool>());
    }
    if constexpr(std::is_same<UserType, std::string>::value) {
      rv = std::make_unique<np::dtype>(p::make_tuple('U', 1));
    }
    else {
      rv = std::make_unique<np::dtype>(np::dtype::get_builtin<UserType>());
    }
  });
  return *rv;
}