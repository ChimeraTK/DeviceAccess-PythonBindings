#ifndef TEST_ACCESSOR_H_
#define TEST_ACCESSOR_H_

#include "register.h"

#include <ChimeraTK/AccessMode.h>
#include <ChimeraTK/NDRegisterAccessor.h>
#include "backend.h"

namespace TestBackend {

  template<typename UserType>
  class TestBackEndAccessor : public ChimeraTK::NDRegisterAccessor<UserType> {
    Register::View _view;
    boost::shared_ptr<Backend> _testBackend;

   public:
    TestBackEndAccessor(Register::View& v, ChimeraTK::AccessModeFlags flags, boost::shared_ptr<Backend> testBackend)
    : ChimeraTK::NDRegisterAccessor<UserType>(registerName(v), flags), _view(v), _testBackend(testBackend) {
      ChimeraTK::TransferElement::_exceptionBackend = testBackend;
      std::set<ChimeraTK::AccessMode> supportedFlags{ChimeraTK::AccessMode::raw, //
          ChimeraTK::AccessMode::wait_for_new_data};

      flags.checkForUnknownFlags(supportedFlags);

      using NDAccessor_t = ChimeraTK::NDRegisterAccessor<UserType>;
      NDAccessor_t::buffer_2D.resize(rows(_view));
      for(auto& e : NDAccessor_t::buffer_2D) {
        e.resize(columns(_view));
      }
    }

    bool isReadOnly() const override;
    bool isReadable() const override;
    bool isWriteable() const override;
    void doReadTransferSynchronously() override;
    bool doWriteTransfer(ChimeraTK::VersionNumber versionNumber = {}) override;
    void doPostRead(ChimeraTK::TransferType, bool hasNewData) override;
    std::list<boost::shared_ptr<ChimeraTK::TransferElement>> getInternalElements() override;
    std::vector<boost::shared_ptr<ChimeraTK::TransferElement>> getHardwareAccessingElements() override;
  };

  template<typename UserType>
  inline bool TestBackEndAccessor<UserType>::isReadOnly() const {
    if(getAccessMode(_view) == TestBackend::Register::Access::ro) {
      return true;
    }
    else {
      return false;
    }
  }

  template<typename UserType>
  inline bool TestBackEndAccessor<UserType>::isReadable() const {
    using Access_t = TestBackend::Register::Access;
    switch(getAccessMode(_view)) {
      case Access_t::ro:
      case Access_t::rw:
        return true;
      default:
        return false;
    }
  }

  template<typename UserType>
  inline bool TestBackEndAccessor<UserType>::isWriteable() const {
    using Access_t = TestBackend::Register::Access;
    switch(getAccessMode(_view)) {
      case Access_t::rw:
      case Access_t::wo:
        return true;
      default:
        return false;
    }
  }

  /***************************************************************************/
  template<typename UserType>
  inline void TestBackEndAccessor<UserType>::doReadTransferSynchronously() {
    // nothing to do here, the actual transfer happens directly to the buffer_2D,
    // so we have to do it in doPostRead().
    if(_testBackend->_hasException) {
      throw ChimeraTK::runtime_error("Previous, unrecovered error in TestBackend.");
    }
  }

  /***************************************************************************/
  template<typename UserType>
  inline void TestBackEndAccessor<UserType>::doPostRead(ChimeraTK::TransferType, bool hasNewData) {
    if(!hasNewData) return;
    using NDAccessor_t = ChimeraTK::NDRegisterAccessor<UserType>;
    NDAccessor_t::buffer_2D = _view.read<UserType>();
    ChimeraTK::TransferElement::_versionNumber = {};
  }
  /***************************************************************************/
  template<typename UserType>
  inline bool TestBackEndAccessor<UserType>::doWriteTransfer([[maybe_unused]] ChimeraTK::VersionNumber versionNumber) {
    using NDAccessor_t = ChimeraTK::NDRegisterAccessor<UserType>;
    if(_testBackend->_hasException) {
      throw ChimeraTK::runtime_error("Previous, unrecovered error in TestBackend.");
    }
    _view.write(NDAccessor_t::buffer_2D);
    return true;
  }

  template<typename UserType>
  inline std::list<boost::shared_ptr<ChimeraTK::TransferElement>> TestBackEndAccessor<UserType>::getInternalElements() {
    return {};
  }

  template<typename UserType>
  inline std::vector<boost::shared_ptr<ChimeraTK::TransferElement>> TestBackEndAccessor<
      UserType>::getHardwareAccessingElements() {
    return {boost::enable_shared_from_this<ChimeraTK::TransferElement>::shared_from_this()};
  }

} // namespace TestBackend
#endif /* TEST_ACCESSOR_H_ */
