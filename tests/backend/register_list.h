#ifndef REGISTER_LIST_H_
#define REGISTER_LIST_H_

#include "new_class.h"
#include "register_access.h"

#include <boost/shared_ptr.hpp>
#include <ChimeraTK/RegisterCatalogue.h>
#include <unordered_map>
#include <vector>

namespace TestBackend {
using RegisterList = std::unordered_map<std::string, Register>;

RegisterList getDummyList();
ChimeraTK::RegisterCatalogue convertToRegisterCatalogue(RegisterList const &l);
Register &search(RegisterList &l, std::string const &id);

} // namespace TestBackend

#endif /* REGISTER_LIST_H_ */
