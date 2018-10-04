#include <boost/variant.hpp>

#include "VariantTypes.h"
#include "new_class.h"

namespace TestBackend {

using Element = boost::variant<IntegralType,      //
                               FloatingPointType, //
                               BooleanType,       //
                               StringType>;
using ElementStore = std::vector<std::vector<Element>>;



} // namespace TestBackend
