#include "PythonExceptions.h"
#include "PythonModuleMethods.h"

#include <ChimeraTK/Device.h>
#include <ChimeraTK/Utilities.h>

namespace mtca4upy {
  namespace DeviceAccess {

    ChimeraTK::TwoDRegisterAccessor<double> getTwoDAccessor(
        const ChimeraTK::Device& self, const std::string& registerPath) {
      return (self.getTwoDRegisterAccessor<double>(registerPath));
    }

    ChimeraTK::OneDRegisterAccessor<int32_t> getRawOneDAccessor(const ChimeraTK::Device& self,
        const std::string& registerPath, size_t numberOfelementsToRead, size_t elementOffset) {
      return self.getOneDRegisterAccessor<int32_t>(
          registerPath, numberOfelementsToRead, elementOffset, {ChimeraTK::AccessMode::raw});
    }

    std::string getCatalogueMetadata(const ChimeraTK::Device& self, const std::string& parameterName) {
      return self.getMetadataCatalogue().getMetadata(parameterName);
    }

    void open(ChimeraTK::Device& self) {
      self.open();
    }
    void open(ChimeraTK::Device& self, std::string const& aliasName) {
      self.open(aliasName);
    }
    void close(ChimeraTK::Device& self) {
      self.close();
    }

  } // namespace DeviceAccess

  namespace TransferElementID {
    bool isValid(ChimeraTK::TransferElementID& self) {
      return self.isValid();
    }
    bool lt(ChimeraTK::TransferElementID& self, ChimeraTK::TransferElementID& other) {
      return self != other && std::less<ChimeraTK::TransferElementID>{}(self, other);
    }
    bool eq(ChimeraTK::TransferElementID& self, ChimeraTK::TransferElementID& other) {
      return self == other;
    }
    bool ge(ChimeraTK::TransferElementID& self, ChimeraTK::TransferElementID& other) {
      return self == other || std::less<ChimeraTK::TransferElementID>{}(other, self);
    }
    bool le(ChimeraTK::TransferElementID& self, ChimeraTK::TransferElementID& other) {
      return self == other || std::less<ChimeraTK::TransferElementID>{}(self, other);
    }
    bool ne(ChimeraTK::TransferElementID& self, ChimeraTK::TransferElementID& other) {
      return self != other;
    }
    bool gt(ChimeraTK::TransferElementID& self, ChimeraTK::TransferElementID& other) {
      return self != other && std::less<ChimeraTK::TransferElementID>{}(other, self);
    }

  } // namespace TransferElementID

  namespace VersionNumber {
    boost::posix_time::ptime getTime([[maybe_unused]] ChimeraTK::VersionNumber& self) {
      return boost::posix_time::ptime(boost::gregorian::date(1990, 1, 1));
    }
    std::string str(ChimeraTK::VersionNumber& self) {
      return std::string(self);
    }
    bool lt(ChimeraTK::VersionNumber& self, ChimeraTK::VersionNumber& other) {
      return self < other;
    }
    bool eq(ChimeraTK::VersionNumber& self, ChimeraTK::VersionNumber& other) {
      return self == other;
    }
    bool ge(ChimeraTK::VersionNumber& self, ChimeraTK::VersionNumber& other) {
      return self >= other;
    }
    bool le(ChimeraTK::VersionNumber& self, ChimeraTK::VersionNumber& other) {
      return self <= other;
    }
    bool ne(ChimeraTK::VersionNumber& self, ChimeraTK::VersionNumber& other) {
      return self != other;
    }
    bool gt(ChimeraTK::VersionNumber& self, ChimeraTK::VersionNumber& other) {
      return self > other;
    }

    ChimeraTK::VersionNumber getNullVersion() {
      return ChimeraTK::VersionNumber(nullptr);
    }

  } // namespace VersionNumber
} // namespace mtca4upy
void mtca4upy::setDmapFile(const std::string& dmapFile) {
  ChimeraTK::setDMapFilePath(dmapFile);
}

std::string mtca4upy::getDmapFile() {
  return (ChimeraTK::getDMapFilePath());
}