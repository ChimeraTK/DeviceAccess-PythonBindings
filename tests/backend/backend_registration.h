#include "backend.h"

#include <ChimeraTK/DeviceBackendImpl.h>
#include <ChimeraTK/BackendFactory.h>
#include <ChimeraTK/DeviceAccessVersion.h>

namespace TestBackend {
using Backend_t = boost::shared_ptr<ChimeraTK::DeviceBackend>;
class BackendRegisterer {
public:
  BackendRegisterer();

private:
  static Backend_t createBackend(std::string /*host*/, std::string /*instance*/,
                                 std::list<std::string> parameters,
                                 std::string /*mapFileName*/);
};

BackendRegisterer::BackendRegisterer() {
  ChimeraTK::BackendFactory::getInstance().registerBackendType(
      "TestBackend", "", &BackendRegisterer::createBackend,
      CHIMERATK_DEVICEACCESS_VERSION);
}
Backend_t BackendRegisterer::createBackend(std::string /*host*/,     //
                                           std::string /*instance*/, //
                                           std::list<std::string>,
                                           std::string /*mapFileName*/) {
  return Backend_t(new Backend(TestBackend::getDummyList()));
}


} // namespace TestBackend

extern TestBackend::BackendRegisterer registerer;
