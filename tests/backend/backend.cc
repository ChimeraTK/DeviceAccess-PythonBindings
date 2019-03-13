#include "backend.h"
#include "test_accessor.h"

#include <ChimeraTK/DeviceAccessVersion.h>
#include <ChimeraTK/VirtualFunctionTemplate.h>

template<typename UserType>
using Accessor_t = boost::shared_ptr<ChimeraTK::NDRegisterAccessor<UserType>>;

extern "C" {
const char* deviceAccessVersionUsedToCompile() {
  return CHIMERATK_DEVICEACCESS_VERSION;
}
}

namespace TestBackend {

  struct Backend::Impl {
    RegisterList list_;
    Impl(RegisterList&& l) : list_(std::move(l)) {}
  };

  /****************************************************************************
   Public Interface of TestBackend:
  */
  Backend::Backend(RegisterList l)              //
  : impl_(std::make_unique<Impl>(std::move(l))) //
  {
    FILL_VIRTUAL_FUNCTION_TEMPLATE_VTABLE(getRegisterAccessor_impl);
    _catalogue = TestBackend::convertToRegisterCatalogue(impl_->list_);
  }

  Backend::~Backend() = default; // to avoid static assert with unique_ptr

  void Backend::open() { _opened = true; }
  void Backend::close() { _opened = false; }

  std::string Backend::readDeviceInfo() {
    return std::string("This backend is intended to test ChimeraTK python bindings");
  }
  /****************************************************************************/
  template<typename UserType>
  Accessor_t<UserType>                                                                   //
      Backend::getRegisterAccessor_impl(const ChimeraTK::RegisterPath& registerPathName, //
          size_t numberOfWords,                                                          //
          size_t wordOffsetInRegister,                                                   //
          ChimeraTK::AccessModeFlags flags)                                              //
  {
    auto& r = search(impl_->list_, registerPathName);

    auto nSequences = rows(r);
    auto view = r.getView({{nSequences, numberOfWords}, //
        0,                                              //
        wordOffsetInRegister});

    return Accessor_t<UserType>(new TestBackEndAccessor<UserType>(view, flags));
  }

} // namespace TestBackend
