#ifndef TEST_ACCESSOR_H_
#define TEST_ACCESSOR_H_

#include <mtca4u/SyncNDRegisterAccessor.h>
#include <ChimeraTK/AccessMode.h>
#include "register_access.h"

namespace TestBackend {

template <typename UserType>
class TestBackEndAccessor : public mtca4u::SyncNDRegisterAccessor<UserType> {

  DBaseElem& elem_;
  std::size_t numWords_;
  std::size_t wordOffset_;

public:
  TestBackEndAccessor() {}
  TestBackEndAccessor(DBaseElem& elem, std::string const& registerPathName,
                      std::size_t numberOfWords,
                      std::size_t wordOffsetInRegister,
                      ChimeraTK::AccessModeFlags flags);
  bool isReadOnly() const override;
  bool isReadable() const override;
  bool isWriteable() const override;
  void doReadTransfer() override;
  bool doReadTransferNonBlocking() override;
  bool doReadTransferLatest() override;
  bool doWriteTransfer(ChimeraTK::VersionNumber versionNumber = {}) override;
  std::list<boost::shared_ptr<mtca4u::TransferElement> > getInternalElements()
      override;
  std::vector<boost::shared_ptr<mtca4u::TransferElement> >
  getHardwareAccessingElements() override;
};

template <typename UserType>
inline bool TestBackEndAccessor<UserType>::isReadOnly() const {
  if (access(elem_) == TestBackend::AccessMode::ro) {
    return true;
  } else {
    return false;
  }
}

template <typename UserType>
TestBackEndAccessor<UserType>::TestBackEndAccessor(
    DBaseElem& elem, std::string const& registerPathName,
    std::size_t numberOfWords, std::size_t wordOffsetInRegister,
    ChimeraTK::AccessModeFlags flags)
    : ChimeraTK::SyncNDRegisterAccessor<UserType>(registerPathName),
      elem_(elem),
      numWords_((numberOfWords == 0) ? elem.getElements() : numberOfWords),
      wordOffset_(wordOffsetInRegister) {

  try {
    std::set<ChimeraTK::AccessMode> supportedFlags{
      ChimeraTK::AccessMode::raw, //
      ChimeraTK::AccessMode::wait_for_new_data
    };

    flags.checkForUnknownFlags(supportedFlags);

    using NDAccessor_t = ChimeraTK::NDRegisterAccessor<UserType>;
    NDAccessor_t::buffer_2D.resize(elem_.getChannels());
    for (auto& e : NDAccessor_t::buffer_2D) {
      e.resize(numWords_);
    }
  }
  catch (...) {
    this->shutdown();
    throw;
  }
}

template <typename UserType>
inline bool TestBackEndAccessor<UserType>::isReadable() const {
  using Access_t = TestBackend::AccessMode;
  switch (access(elem_)) {
    case Access_t::ro:
    case Access_t::rw:
      return true;
    default:
      return false;
  }
}

template <typename UserType>
inline bool TestBackEndAccessor<UserType>::isWriteable() const {
  using Access_t = TestBackend::AccessMode;
  switch (access(elem_)) {
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
  auto x_offset = wordOffset_;
  auto y_offset = 0;
  copyFrom(elem_, NDAccessor_t::buffer_2D, x_offset, y_offset);
}

template <typename UserType>
inline bool TestBackEndAccessor<UserType>::doWriteTransfer(
    ChimeraTK::VersionNumber /*versionNumber*/) {
  using NDAccessor_t = ChimeraTK::NDRegisterAccessor<UserType>;
  auto x_offset = wordOffset_;
  auto y_offset = 0;

  std::size_t channelsWritten;
  std::size_t elementsWritten;

  std::tie(elementsWritten, channelsWritten) =
      copyInto(elem_, //
               NDAccessor_t::buffer_2D, x_offset, y_offset);

  if ((channelsWritten == elem_.getChannels()) &&
      (elementsWritten == numWords_)) {
    return false; // false == succsess?
  } else {
    // TODO: replace with proper exception
    throw std::runtime_error("Write to backend register failed");
  }
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
inline std::list<boost::shared_ptr<mtca4u::TransferElement> >
TestBackEndAccessor<UserType>::getInternalElements() {
  return {};
}

template <typename UserType>
inline std::vector<boost::shared_ptr<mtca4u::TransferElement> >
TestBackEndAccessor<UserType>::getHardwareAccessingElements() {
  return { boost::enable_shared_from_this<
      ChimeraTK::TransferElement>::shared_from_this() };
}

} // namespace TestBackend
#endif /* TEST_ACCESSOR_H_ */
