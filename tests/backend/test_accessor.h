#ifndef TEST_ACCESSOR_H_
#define TEST_ACCESSOR_H_

#include "register.h"

#include <ChimeraTK/AccessMode.h>
#include <ChimeraTK/SyncNDRegisterAccessor.h>

namespace TestBackend {

template <typename UserType>
class TestBackEndAccessor : public ChimeraTK::SyncNDRegisterAccessor<UserType> {

  Register::View view_;

public:
  TestBackEndAccessor(Register::View &v, ChimeraTK::AccessModeFlags flags)
      : ChimeraTK::SyncNDRegisterAccessor<UserType>(registerName(v)), view_(v) {

    try {
      std::set<ChimeraTK::AccessMode> supportedFlags{
          ChimeraTK::AccessMode::raw, //
          ChimeraTK::AccessMode::wait_for_new_data};

      flags.checkForUnknownFlags(supportedFlags);

      using NDAccessor_t = ChimeraTK::NDRegisterAccessor<UserType>;
      NDAccessor_t::buffer_2D.resize(columns(view_));
      for (auto &e : NDAccessor_t::buffer_2D) {
        e.resize(rows(view_));
      }
    } catch (...) {
      this->shutdown();
      throw;
    }
  }

  virtual ~TestBackEndAccessor() {
    ChimeraTK::SyncNDRegisterAccessor<UserType>::shutdown();
  }
  bool isReadOnly() const override;
  bool isReadable() const override;
  bool isWriteable() const override;
  void doReadTransfer() override;
  bool doReadTransferNonBlocking() override;
  bool doReadTransferLatest() override;
  bool doWriteTransfer(ChimeraTK::VersionNumber versionNumber = {}) override;
  std::list<boost::shared_ptr<ChimeraTK::TransferElement>>
  getInternalElements() override;
  std::vector<boost::shared_ptr<ChimeraTK::TransferElement>>
  getHardwareAccessingElements() override;
  ChimeraTK::AccessModeFlags getAccessModeFlags() const override {
    // TODO
  }
};

template <typename UserType>
inline bool TestBackEndAccessor<UserType>::isReadOnly() const {
  if (getAccess(view_) == TestBackend::Register::Access::ro) {
    return true;
  } else {
    return false;
  }
}
/*template <typename UserType> ChimeraTK::AccessModeFlags getAccessModeFlags() {
  return {};
}*/

template <typename UserType>
inline bool TestBackEndAccessor<UserType>::isReadable() const {
  using Access_t = TestBackend::Register::Access;
  switch (getAccess(view_)) {
  case Access_t::ro:
  case Access_t::rw:
    return true;
  default:
    return false;
  }
}

template <typename UserType>
inline bool TestBackEndAccessor<UserType>::isWriteable() const {
  using Access_t = TestBackend::Register::Access;
  switch (getAccess(view_)) {
  case Access_t::rw:
  case Access_t::wo:
    return true;
  default:
    return false;
  }
}

/***************************************************************************/
template <typename UserType>
inline void TestBackEndAccessor<UserType>::doReadTransfer() {
  using NDAccessor_t = ChimeraTK::NDRegisterAccessor<UserType>;
  NDAccessor_t::buffer_2D = view_.read<UserType>();
}
/***************************************************************************/
template <typename UserType>
inline bool TestBackEndAccessor<UserType>::doWriteTransfer(
    ChimeraTK::VersionNumber /*versionNumber*/) {
  using NDAccessor_t = ChimeraTK::NDRegisterAccessor<UserType>;
  view_.write(NDAccessor_t::buffer_2D);
  return true;
}
/***************************************************************************/
template <typename UserType>
inline bool TestBackEndAccessor<UserType>::doReadTransferNonBlocking() {
  // auto data  = dbase.get<UserType>(path); //iterator
  // TODO: flesh this out later when we start to deal with async and push types.
  doReadTransfer();
  return true;
}

template <typename UserType>
inline bool TestBackEndAccessor<UserType>::doReadTransferLatest() {
  // TODO: flesh this out later when we start to deal with async and push types.
  doReadTransfer();
  return true;
}

template <typename UserType>
inline std::list<boost::shared_ptr<ChimeraTK::TransferElement>>
TestBackEndAccessor<UserType>::getInternalElements() {
  return {};
}

template <typename UserType>
inline std::vector<boost::shared_ptr<ChimeraTK::TransferElement>>
TestBackEndAccessor<UserType>::getHardwareAccessingElements() {
  return {boost::enable_shared_from_this<
      ChimeraTK::TransferElement>::shared_from_this()};
}

} // namespace TestBackend
#endif /* TEST_ACCESSOR_H_ */
