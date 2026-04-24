// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <pybind11/pybind11.h>
// pybind11.h must come first

#include "PyDataConsistencyGroup.h"
#include "PyDataType.h"
#include "PyDevice.h"
#include "PyOneDRegisterAccessor.h"
#include "PyReadAnyGroup.h"
#include "PyTransferGroup.h"
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
  ChimeraTK::PyReadAnyGroup::bind(m);
  ChimeraTK::PyReadAnyGroupNotification::bind(m);
  ChimeraTK::PyDataConsistencyGroup::bind(m);
  ChimeraTK::PyMatchingMode::bind(m);
  ChimeraTK::PyTransferGroup::bind(m);

  m.def("setDMapFilePath", ChimeraTK::setDMapFilePath, py::arg("dmapFilePath"),
      R"(Set the location of the dmap file.

      Args:
        dmapFilePath (str): Relative or absolute path of the dmap file (directory and file name).

      Returns:
        None: This function does not return a value.)");

  m.def("getDMapFilePath", ChimeraTK::getDMapFilePath,
      R"(Return the dmap file name which the library currently uses for looking up device(alias) names.

      Returns:
        str: Path of the dmap file (directory and file name).)");

  py::enum_<ChimeraTK::AccessMode>(m, "AccessMode",
      R"(Access mode flags for register access.

      Note:
        Using the raw flag makes code dependent on the backend type, since the actual raw data type must be known.)")
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
      .value("ok", ChimeraTK::DataValidity::ok, "The data is considered valid.")
      .value("faulty", ChimeraTK::DataValidity::faulty, "The data is not considered valid.")
      .export_values();

  py::class_<ChimeraTK::TransferElementID>(m, "TransferElementID")
      .def("isValid", &ChimeraTK::TransferElementID::isValid,
          R"(Check whether the ID is valid.

        Returns:
          bool: `True if the ID is valid, `False` otherwise.)")
      .def("__ne__", &ChimeraTK::TransferElementID::operator!=, py::arg("other"),
          R"(Compare two TransferElement IDs for inequality.

        Args:
          other (TransferElementID): ID to compare with.

        Returns:
          bool: True if the IDs are different, false otherwise.)")
      .def(
          "__hash__",
          [](const ChimeraTK::TransferElementID& self) { return std::hash<ChimeraTK::TransferElementID>{}(self); },
          R"(Return the hash value of this TransferElement ID.

        Returns:
          int: Hash value.)")
      .def("__eq__", &ChimeraTK::TransferElementID::operator==, py::arg("other"),
          R"(Compare two TransferElement IDs for equality.

        Args:
          other (TransferElementID): ID to compare with.

        Returns:
          bool: True if the IDs are equal, false otherwise.)");

  py::class_<ChimeraTK::RegisterPath>(m, "RegisterPath",
      R"(Class to store a register path name.

      Elements of the path are separated by a "/" character, but an alternative separator character such as "."
      can optionally be specified as well. Different equivalent notations are converted into a standardised notation
      automatically.)")
      .def(py::init<ChimeraTK::RegisterPath>(), py::arg("other"),
          R"(Create a RegisterPath by copying another RegisterPath.

        Args:
          other (RegisterPath): Path to copy.)")
      .def(py::init<const std::string&>(), py::arg("path"),
          R"(Create a RegisterPath from a path string.

        Args:
          path (str): Register path string.)")
      .def("__str__", &ChimeraTK::RegisterPath::operator std::string,
          R"(Return the path as a string.

        Returns:
          str: Register path string.)")
      .def("setAltSeparator", &ChimeraTK::RegisterPath::setAltSeparator, py::arg("altSeparator"),
          R"(Set alternative separator.

        Args:
          altSeparator (str): Alternative separator character to use instead of "/". Use an empty string to reset to default.

        Returns:
          None: This function does not return a value.)")
      .def("getWithAltSeparator", &ChimeraTK::RegisterPath::getWithAltSeparator,
          R"(Obtain path with alternative separator character instead of "/". The leading separator will be omitted.

        Returns:
            str: Register path with alternative separator.)")
      .def("__itruediv__", &ChimeraTK::RegisterPath::operator/=, py::arg("rightHandSide"),
          R"(Modify this object by adding a new element to this path.

        Args:
            rightHandSide (str): New element to add to the path.

        Returns:
            RegisterPath: Modified RegisterPath object.)")
      .def("__iadd__", &ChimeraTK::RegisterPath::operator+=, py::arg("rightHandSide"),
          R"(Modify this object by concatenating the given string to the path.

        Args:
            rightHandSide (str): String to concatenate to the path.

        Returns:
            RegisterPath: Modified RegisterPath object.)")
      .def("__lt__", &ChimeraTK::RegisterPath::operator<, py::arg("other"),
          R"(Compare two paths lexicographically.

        Args:
          other (RegisterPath): Path to compare with.

        Returns:
          bool: True if this path is lexicographically smaller, false otherwise.)")
      .def("length", &ChimeraTK::RegisterPath::length,
          R"(Get the length of the path (including leading slash).

        Returns:
            int: Length of the register path.)")
      .def("startsWith", &ChimeraTK::RegisterPath::startsWith, py::arg("prefix"),
          R"(Check whether the path starts with the given prefix.

        Args:
          prefix (RegisterPath): Prefix to check.

        Returns:
          bool: True if this path starts with prefix, false otherwise.)")
      .def("endsWith", &ChimeraTK::RegisterPath::endsWith, py::arg("suffix"),
          R"(Check whether the path ends with the given suffix.

        Args:
          suffix (RegisterPath): Suffix to check.

        Returns:
          bool: True if this path ends with suffix, false otherwise.)")
      .def("getComponents", &ChimeraTK::RegisterPath::getComponents,
          R"(Split path into components.

        Returns:
            list[str]: List of path components.)")
      .def(
          "__ne__",
          [](const ChimeraTK::RegisterPath& self, const ChimeraTK::RegisterPath& other) { return self != other; },
          py::arg("other"),
          R"(Compare two RegisterPath objects for inequality.

        Args:
          other (RegisterPath): Path to compare with.

        Returns:
          bool: True if the paths are different, false otherwise.)")
      .def(
          "__ne__", [](const ChimeraTK::RegisterPath& self, const std::string& other) { return self != other; },
          py::arg("other"),
          R"(Compare RegisterPath and string for inequality.

        Args:
          other (str): Path string to compare with.

        Returns:
          bool: True if the paths are different, false otherwise.)")
      .def(
          "__eq__",
          [](const ChimeraTK::RegisterPath& self, const ChimeraTK::RegisterPath& other) { return self == other; },
          py::arg("other"),
          R"(Compare two RegisterPath objects for equality.

        Args:
          other (RegisterPath): Path to compare with.

        Returns:
          bool: True if the paths are equal, false otherwise.)")
      .def(
          "__eq__", [](const ChimeraTK::RegisterPath& self, const std::string& other) { return self == other; },
          py::arg("other"),
          R"(Compare RegisterPath and string for equality.

        Args:
          other (str): Path string to compare with.

        Returns:
          bool: True if the paths are equal, false otherwise.)");

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
