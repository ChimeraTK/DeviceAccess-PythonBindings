#include <mtca4u/DeviceBackendImpl.h>
#include <mtca4u/BackendFactory.h>
#include <mtca4u/DeviceAccessVersion.h>

#include "backend.h"

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
  mtca4u::BackendFactory::getInstance().registerBackendType(
      "TestBackend", "", &BackendRegisterer::createBackend,
      CHIMERATK_DEVICEACCESS_VERSION);
}
Backend_t BackendRegisterer::createBackend(std::string /*host*/,     //
                                           std::string /*instance*/, //
                                           std::list<std::string>,
                                           std::string /*mapFileName*/) {
  return Backend_t(new Backend(TestBackend::getRegisterList()));
}

// Invoke registration on loading module.
static auto registerer = BackendRegisterer();
} // namespace TestBackend
