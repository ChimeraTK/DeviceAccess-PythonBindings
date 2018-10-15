/*
 * register_list.h
 *
 *  Created on: Feb 22, 2018
 *      Author: varghese
 */

#ifndef REGISTER_LIST_H_
#define REGISTER_LIST_H_

#include "register_access.h"
#include "new_class.h"
#include <ChimeraTK/RegisterCatalogue.h>
#include <vector>
#include <unordered_map>

namespace TestBackend {

using RegisterList = std::vector<DBaseElem>;
using List = std::unordered_map<std::string, Register>;

List getList();
RegisterList getDummyList();
ChimeraTK::RegisterCatalogue convertToRegisterCatalogue(RegisterList const &l);
ChimeraTK::RegisterCatalogue convertToRegisterCatalogue(List const &l);
DBaseElem &search(RegisterList &l, std::string const &id);
//Register &search(RegisterList &l, std::string const &id);

class TBRegisterInfo : public ChimeraTK::RegisterInfo {
  ChimeraTK::RegisterPath name_;
  std::size_t elements_;
  std::size_t channels_;
  std::size_t dimensions_;
  DataDescriptor descriptor_;

public:
  TBRegisterInfo(ChimeraTK::RegisterPath name, //
                 std::size_t elements,      //
                 std::size_t channels,      //
                 std::size_t dimensions,    //
                 ChimeraTK::RegisterInfo::DataDescriptor descriptor)
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

  ChimeraTK::RegisterPath getRegisterName() const override { return name_; }
  unsigned int getNumberOfElements() const override { return elements_; }
  unsigned int getNumberOfChannels() const override { return channels_; }
  unsigned int getNumberOfDimensions() const override { return dimensions_; }
  DataDescriptor const& getDataDescriptor() const override {
    return descriptor_;
  }
};
} // namespace TestBackend

#endif /* REGISTER_LIST_H_ */
