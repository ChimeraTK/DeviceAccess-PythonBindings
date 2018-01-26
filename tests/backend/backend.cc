#include "backend.h"

#include <ChimeraTK/VirtualFunctionTemplate.h>
#include <ChimeraTK/DeviceAccessVersion.h>

#include "test_accessor.h"
#include "register_list.h"
#include "test_accessor.h"

template <typename UserType>
using Accessor_t = boost::shared_ptr<ChimeraTK::NDRegisterAccessor<UserType> >;

extern "C" {
const char* deviceAccessVersionUsedToCompile() {
  return CHIMERATK_DEVICEACCESS_VERSION;
}
}

namespace TestBackend {

struct Backend::Impl {
  RegisterList list_;
  Impl(RegisterList l) : list_(std::move(l)) {}
};

/*****************************************************************************
 * Defines functor class accessorFactory_vtable_filler */
DEFINE_VIRTUAL_FUNCTION_TEMPLATE_VTABLE_FILLER(Backend, accessorFactory, 4);

/****************************************************************************
 Public Interface of TestBackend:
*/

Backend::Backend(RegisterList l) : impl_(std::make_unique<Impl>(std::move(l))) {
  _catalogue = TestBackend::getRegisterCatalogue(impl_->list_);
  std::sort(impl_->list_.begin(), impl_->list_.end());
  boost::fusion::for_each(getRegisterAccessor_impl_vtable.table,
                          accessorFactory_vtable_filler(this));
}

Backend::~Backend() = default; // to avoid static assert with unique_ptr

void Backend::open() { _opened = true; }
void Backend::close() { _opened = false; }
std::string Backend::readDeviceInfo() {
  return std::string("This backend used to test ChimeraTK python bindings");
}

/****************************************************************************/

template <typename UserType>
Accessor_t<UserType>                                                          //
    Backend::accessorFactory(const ChimeraTK::RegisterPath& registerPathName, //
                             size_t numberOfWords,                            //
                             size_t wordOffsetInRegister,                     //
                             ChimeraTK::AccessModeFlags flags) {
  DBaseElem& elem = search(impl_->list_, registerPathName);
  return Accessor_t<UserType>(
      new TestBackEndAccessor<UserType>(elem,                 //
                                        registerPathName,     //
                                        numberOfWords,        //
                                        wordOffsetInRegister, //
                                        flags));
}

} // namespace TestBackend
  // TODO: define the filler class
