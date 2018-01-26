#include "register_access.h"

// emulate the catalogue
// DBASE is a
// - list_of_dbase_elements
// - None

// compound type DBaseElement
// - std::string pathname
// - Data
// - Access type

// Data is a compund type
// - std::vector<T>
// - type
//     - shape (channels, elements)
//     - dimension
//     - channels
//     - elements

/*
 * - Make up a DBASE from DBASE elements
 * - Access memory of DBASE_ELEMENTS for data
 * - Query attributes of DBASE_ELEMENT
 *     Atributes include:
 *     - AccessType
 *     - type
 *     - shape (channels, elements)
 *     - dimension
 *     - channels
 *     - elements
 * - Collect and give out attribute list in DBASE as a register catalogue
*/

/*
 * PARSING
 * - How feasible is creating DBAse from Register catalogue.
 *
 */

namespace TestBackend {

bool isSubShape(DBaseElem const& e,   //
                std::size_t x_size,   //
                std::size_t y_size,   //
                std::size_t x_offset, //
                std::size_t y_offset) {
  std::size_t e_x_size;
  std::size_t e_y_size;
  std::tie(e_x_size, e_y_size) = shape(e);

  if ((x_offset + x_size > e_x_size) || ((y_offset + y_size) > e_y_size)) {
    return false;
  } else {
    return true;
  }
}

std::tuple<std::size_t, std::size_t> //
    shape(const DBaseElem& e) {
  auto vis = GetShape();
  return boost::apply_visitor(vis, e.value_);
}

AccessMode access(DBaseElem const& e) { return e.access_; }

ElementType type(DBaseElem const& e) {
  auto visitor = GetType();
  return boost::apply_visitor(visitor, e.value_);
}

std::string id(DBaseElem const& e) { return e.name_; }

std::size_t DBaseElem::getChannels() const {
  auto vis = GetChannels();
  return boost::apply_visitor(vis, value_);
}
std::size_t DBaseElem::getElements() const {
  auto vis = GetSequences();
  return boost::apply_visitor(vis, value_);
}

std::size_t DBaseElem::getDimensions() const {
  auto channels = getChannels();
  if (channels < 2) {
    return channels;
  } else {
    return 2;
  }
}

} // namespace TestBackend
