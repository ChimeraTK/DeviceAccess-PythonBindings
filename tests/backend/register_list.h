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

namespace TestBackend {

using RegisterList = std::vector<DBaseElem>;

RegisterList getDummyList();
ChimeraTK::RegisterCatalogue convertToRegisterCatalogue(RegisterList const &l);
DBaseElem &search(RegisterList &l, std::string const &id);
//Register &search(RegisterList &l, std::string const &id);

} // namespace TestBackend

#endif /* REGISTER_LIST_H_ */
