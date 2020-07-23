#ifndef TEST_BACKEND_H
#define TEST_BACKEND_H

#include "register_list.h"
#include <ChimeraTK/DeviceBackendImpl.h>

namespace TestBackend {
  class accessorFactory_vtable_filler;
  class Backend : public ChimeraTK::DeviceBackendImpl {
   public:
    Backend(RegisterList l);
    ~Backend() override;

    void open() override;
    void close() override;
    std::string readDeviceInfo() override;
    bool isFunctional() const override;
    void setException() override;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    template<typename UserType>
    boost::shared_ptr<ChimeraTK::NDRegisterAccessor<UserType>>                    //
        getRegisterAccessor_impl(const ChimeraTK::RegisterPath& registerPathName, //
            size_t numberOfWords,                                                 //
            size_t wordOffsetInRegister,                                          //
            ChimeraTK::AccessModeFlags flags);

    bool _hasException{true};

    template<typename UserType>
    friend class TestBackEndAccessor;
  };

} // namespace TestBackend

#endif
