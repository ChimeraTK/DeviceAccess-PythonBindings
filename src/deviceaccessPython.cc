// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <pybind11/pybind11.h>
// pybind11.h must come first

#include "PyDataType.h"
#include "PyDevice.h"
#include "PyOneDRegisterAccessor.h"
#include "PyTwoDRegisterAccessor.h"
#include "PyVersionNumber.h"
#include "RegisterCatalogue.h"

#include <ChimeraTK/SupportedUserTypes.h>
#include <ChimeraTK/VariantUserTypes.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

//****************************************************************************//

// Holder definitions
PYBIND11_DECLARE_HOLDER_TYPE(T, boost::shared_ptr<T>)

//****************************************************************************//

PYBIND11_MODULE(deviceaccess, m) {
  ChimeraTK::PyDevice::bind(m);
  ChimeraTK::PyVersionNumber::bind(m);
  ChimeraTK::PyTransferElementBase::bind(m);
  ChimeraTK::PyScalarRegisterAccessor::bind(m);
  ChimeraTK::PyTwoDRegisterAccessor::bind(m);
  ChimeraTK::PyOneDRegisterAccessor::bind(m);
  ChimeraTK::PyVoidRegisterAccessor::bind(m);
  ChimeraTK::PyDataType::bind(m);
  DeviceAccessPython::RegisterCatalogue::bind(m);
  DeviceAccessPython::RegisterInfo::bind(m);
  DeviceAccessPython::RegisterInfo::bindBackendRegisterInfoBase(m);
  DeviceAccessPython::DataDescriptor::bind(m);

  m.def("setDMapFilePath", ChimeraTK::setDMapFilePath, py::arg("dmapFilePath"),
      R"(Set the location of the dmap file.

        :param dmapFilePath: Relative or absolute path of the dmap file (directory and file name).
        :type dmapFilePath: str)");

  m.def("getDMapFilePath", ChimeraTK::getDMapFilePath,
      R"(Returns the dmap file name which the library currently uses for looking up device(alias) names.

        :return: Path of the dmap file (directory and file name).
        :rtype: str)");

  py::enum_<ChimeraTK::AccessMode>(m, "AccessMode",
      R"(Access mode flags for register access.

        Note:
            Using the raw flag will make your code intrinsically dependent on the backend type, since the actual raw data type must be known.)")
      .value("raw", ChimeraTK::AccessMode::raw,
          R"(This access mode disables any possible conversion from the original hardware data type into the given UserType. Obtaining the accessor with a UserType unequal to the actual raw data type will fail and throw an exception.)")
      .value("wait_for_new_data", ChimeraTK::AccessMode::wait_for_new_data,
          R"(This access mode makes any read blocking until new data has arrived since the last read. This flag may not be supported by all registers (and backends), in which case an exception will be thrown.)")
      .export_values();

  py::enum_<ChimeraTK::DataDescriptor::FundamentalType>(m, "FundamentalType",
      "This is only used inside the DataDescriptor class; defined outside to prevent too long fully qualified names.")
      .value("numeric", ChimeraTK::DataDescriptor::FundamentalType::numeric)
      .value("string", ChimeraTK::DataDescriptor::FundamentalType::string)
      .value("boolean", ChimeraTK::DataDescriptor::FundamentalType::boolean)
      .value("nodata", ChimeraTK::DataDescriptor::FundamentalType::nodata)
      .value("undefined", ChimeraTK::DataDescriptor::FundamentalType::undefined)
      .export_values();

  py::enum_<ChimeraTK::DataValidity>(m, "DataValidity",
      R"(The current state of the data.

        Note:
            This is a flag to describe the validity of the data. It should be used to signalize
            whether or not to trust the data currently. It MUST NOT be used to signalize any
            communication errors with a device, rather to signalize the consumer after such an
            error that the data is currently not trustable, because we are performing calculations
            with the last known valid data, for example.)")
      .value("ok", ChimeraTK::DataValidity::ok, "The data is considered valid")
      .value("faulty", ChimeraTK::DataValidity::faulty, "The data is not considered valid")
      .export_values();

  py::class_<ChimeraTK::TransferElementID>(m, "TransferElementID")
      .def("isValid", &ChimeraTK::TransferElementID::isValid, "Check whether the ID is valid.")
      .def("__ne__", &ChimeraTK::TransferElementID::operator!=)
      .def("__eq__", &ChimeraTK::TransferElementID::operator==);

  py::class_<ChimeraTK::RegisterPath>(m, "RegisterPath",
      R"a(Class to store a register path name. Elements of the path are separated by a "/" character, but an  separation character (e.g. ".") can optionally be specified as well. Different equivalent notations will be converted into a standardised notation automatically.)a")
      .def(py::init<ChimeraTK::RegisterPath>())
      .def(py::init<const std::string&>(), py::arg("path"))
      .def("__str__", &ChimeraTK::RegisterPath::operator std::string)
      .def("setAltSeparator", &ChimeraTK::RegisterPath::setAltSeparator, py::arg("altSeparator"),
          R"(Set alternative separator.

        :param altSeparator: Alternative separator character to use instead of "/". Use an empty string to reset to default.
        :type altSeparator: str)")
      .def("getWithAltSeparator", &ChimeraTK::RegisterPath::getWithAltSeparator,
          R"(Obtain path with alternative separator character instead of "/". The leading separator will be omitted.

        :return: Register path with alternative separator.
        :rtype: str)")
      .def("__itruediv__", &ChimeraTK::RegisterPath::operator/=, py::arg("rightHandSide"),
          R"(Modify this object by adding a new element to this path.

        :param rightHandSide: New element to add to the path.
        :type rightHandSide: str

        :return: Modified RegisterPath object.
        :rtype: RegisterPath)")
      .def("__iadd__", &ChimeraTK::RegisterPath::operator+=, py::arg("rightHandSide"),
          R"(Modify this object by concatenating the given string to the path.

        :param rightHandSide: String to concatenate to the path.
        :type rightHandSide: str

        :return: Modified RegisterPath object.
        :rtype: RegisterPath)")
      .def("__lt__", &ChimeraTK::RegisterPath::operator<)
      .def("length", &ChimeraTK::RegisterPath::length,
          R"(Get the length of the path (including leading slash).

        :return: Length of the register path.
        :rtype: int)")
      .def("startsWith", &ChimeraTK::RegisterPath::startsWith)
      .def("endsWith", &ChimeraTK::RegisterPath::endsWith)
      .def("getComponents", &ChimeraTK::RegisterPath::getComponents,
          R"(Split path into components.

        :return: list of path components.
        :rtype: list[str])")
      .def("__ne__",
          [](const ChimeraTK::RegisterPath& self, const ChimeraTK::RegisterPath& other) { return self != other; })
      .def("__ne__", [](const ChimeraTK::RegisterPath& self, const std::string& other) { return self != other; })
      .def("__eq__",
          [](const ChimeraTK::RegisterPath& self, const ChimeraTK::RegisterPath& other) { return self == other; })
      .def("__eq__", [](const ChimeraTK::RegisterPath& self, const std::string& other) { return self == other; });

  py::implicitly_convertible<std::string, ChimeraTK::RegisterPath>();

  // Map the boost::thread_interrupted exception. We cannot use py::register_exception because there is no what().
  static py::exception<boost::thread_interrupted> exc(m, "ThreadInterrupted");
  // NOLINTNEXTLINE(performance-unnecessary-value-param) - signature with reference not accepted here...
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if(p) {
        std::rethrow_exception(p);
      }
    }
    catch(const boost::thread_interrupted&) {
      exc("Thread Interrupted");
    }
  });
}
