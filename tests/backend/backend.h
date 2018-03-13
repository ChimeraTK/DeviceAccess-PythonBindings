#ifndef TEST_BACKEND_H
#define TEST_BACKEND_H

#include <mtca4u/DeviceBackendImpl.h>
#include "register_list.h"

namespace TestBackend {
class accessorFactory_vtable_filler;
class Backend : public ChimeraTK::DeviceBackendImpl {
public:
  Backend(RegisterList l);
  virtual ~Backend();
  void open() override;
  void close() override;
  std::string readDeviceInfo() override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend accessorFactory_vtable_filler;
  template <typename UserType>
  boost::shared_ptr<ChimeraTK::NDRegisterAccessor<UserType> >          //
      accessorFactory(const ChimeraTK::RegisterPath &registerPathName, //
                      size_t numberOfWords,                            //
                      size_t wordOffsetInRegister,                     //
                      ChimeraTK::AccessModeFlags flags);
};

} // namespace TestBackend

#endif
