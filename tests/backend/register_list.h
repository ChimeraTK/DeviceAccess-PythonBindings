/*
 * register_list.h
 *
 *  Created on: Feb 22, 2018
 *      Author: varghese
 */

#ifndef REGISTER_LIST_H_
#define REGISTER_LIST_H_

#include <vector>
#include <ChimeraTK/RegisterCatalogue.h>
#include "register_access.h"

namespace TestBackend {

using RegisterList = std::vector<DBaseElem>;

RegisterList getRegisterList();
ChimeraTK::RegisterCatalogue getRegisterCatalogue(RegisterList const& l);
DBaseElem& search(RegisterList & l, std::string const& id);

} // namespace TestBackend

#endif /* REGISTER_LIST_H_ */
