// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "RegisterCatalogue.h"

#include <pybind11/pybind11.h>
namespace ctk = ChimeraTK;

namespace py = pybind11;

namespace DeviceAccessPython {

  /*******************************************************************************************************************/

  py::list RegisterCatalogue::items(ChimeraTK::RegisterCatalogue& self) {
    py::list registerInfos{};
    for(const auto& regInfo : self) {
      registerInfos.append(ChimeraTK::RegisterInfo(regInfo.clone()));
    }
    return registerInfos;
  }

  /*******************************************************************************************************************/

  py::list RegisterCatalogue::hiddenRegisters(ChimeraTK::RegisterCatalogue& self) {
    py::list registerInfos{};
    for(const auto& regInfo : self.hiddenRegisters()) {
      registerInfos.append(ChimeraTK::RegisterInfo(regInfo.clone()));
    }
    return registerInfos;
  }

  /*******************************************************************************************************************/

  void RegisterCatalogue::bind(py::module& m) {
    py::class_<ChimeraTK::RegisterCatalogue>(m, "RegisterCatalogue")
        .def(py::init<ChimeraTK::RegisterCatalogue>(), "Catalogue of register information.")
        .def(
            "__iter__", [](const ChimeraTK::RegisterCatalogue& s) { return py::make_iterator(s.begin(), s.end()); },
            py::keep_alive<0, 1>(),
            R"(Return an iterator over visible registers in the catalogue.

            Returns:
                iterator: Iterator yielding RegisterInfo objects.)")
        .def("_items", DeviceAccessPython::RegisterCatalogue::items,
            R"(Return all visible registers in the catalogue as a list.

            Returns:
                list[deviceaccess.RegisterInfo]: List of visible RegisterInfo objects.)")
        .def("hiddenRegisters", DeviceAccessPython::RegisterCatalogue::hiddenRegisters,
            R"(Return list of all hidden registers in the catalogue.

            Returns:
                list[deviceaccess.RegisterInfo]: A list of hidden RegisterInfo objects.)")
        .def("hasRegister", &ChimeraTK::RegisterCatalogue::hasRegister, py::arg("registerPathName"),
            R"(Check if register with the given path name exists.

        Args:
            registerPathName (str): Full path name of the register.

        Returns:
            bool: True if register exists in the catalogue, false otherwise.)")
        .def("getNumberOfRegisters", &ChimeraTK::RegisterCatalogue::getNumberOfRegisters,
            R"(Get number of registers in the catalogue.

        Returns:
            int: Number of registers in the catalogue.)")
        .def("getRegister", &ChimeraTK::RegisterCatalogue::getRegister, py::arg("registerPathName"),
            R"(Get register information for a given full path name.

        Args:
            registerPathName (str): Full path name of the register.

        Returns:
            RegisterInfo: Register information.

        Raises:
            ChimeraTK::logic_error: If register does not exist in the catalogue.)");
  }

  /*******************************************************************************************************************/

  ChimeraTK::DataDescriptor RegisterInfo::getDataDescriptor(ChimeraTK::RegisterInfo& self) {
    return self.getDataDescriptor();
  }

  /*******************************************************************************************************************/

  std::string RegisterInfo::getRegisterName(ChimeraTK::RegisterInfo& self) {
    return self.getRegisterName();
  }

  /*******************************************************************************************************************/

  py::list RegisterInfo::getSupportedAccessModes(ChimeraTK::RegisterInfo& self) {
    ChimeraTK::AccessModeFlags flags = self.getSupportedAccessModes();
    py::list python_flags{};
    if(flags.has(ChimeraTK::AccessMode::raw)) python_flags.append(ChimeraTK::AccessMode::raw);
    if(flags.has(ChimeraTK::AccessMode::wait_for_new_data))
      python_flags.append(ChimeraTK::AccessMode::wait_for_new_data);
    return python_flags;
  }

  void RegisterInfo::bind(py::module& m) {
    py::class_<ChimeraTK::RegisterInfo>(m, "RegisterInfo")
        .def(py::init<ChimeraTK::RegisterInfo>(), "Catalogue of register information.")
        .def("getDataDescriptor", DeviceAccessPython::RegisterInfo::getDataDescriptor,
            R"(Return description of the actual payload data for this register.

        Returns:
            DataDescriptor: Object containing information about the data format.)")
        .def("isReadable", &ChimeraTK::RegisterInfo::isReadable,
            R"(Check whether the register is readable.

        Returns:
            bool: True if the register is readable, false otherwise.)")
        .def("isValid", &ChimeraTK::RegisterInfo::isValid,
            R"(Check whether the RegisterInfo object is valid.

        Returns:
            bool: True if the object contains a valid implementation, false otherwise.)")
        .def("isWriteable", &ChimeraTK::RegisterInfo::isWriteable,
            R"(Check whether the register is writeable.

        Returns:
            bool: True if the register is writeable, false otherwise.)")
        .def("getRegisterName", DeviceAccessPython::RegisterInfo::getRegisterName,
            R"(Return the full path name of the register.

        Returns:
            RegisterPath: Full path name of the register (including modules).)")
        .def("getSupportedAccessModes", DeviceAccessPython::RegisterInfo::getSupportedAccessModes,
            R"(Return all supported AccessModes for this register.

        Returns:
            list[AccessMode]: Flags indicating supported access modes.)")
        .def("getNumberOfElements", &ChimeraTK::RegisterInfo::getNumberOfElements,
            R"(Return the number of elements per channel.

        Returns:
            int: Number of elements per channel.)")
        .def("getNumberOfDimensions", &ChimeraTK::RegisterInfo::getNumberOfDimensions,
            R"(Return the number of dimensions of this register.

        Returns:
            int: Number of dimensions (0=scalar, 1=1D array, 2=2D array).)")
        .def("getNumberOfChannels", &ChimeraTK::RegisterInfo::getNumberOfChannels,
            R"(Return the number of channels in the register.

        Returns:
            int: Number of channels.)")
        .def("getQualifiedAsyncId", &ChimeraTK::RegisterInfo::getQualifiedAsyncId,
            R"(Get the fully qualified async::SubDomain ID.

        If the register does not support wait_for_new_data it will be empty.
        Note: At the moment using async::Domain and async::SubDomain is not mandatory yet, so the ID might be empty even if the register supports wait_for_new_data.

        Returns:
            list[int]: List of IDs forming the fully qualified async::SubDomain ID.)")
        .def("getTags", &ChimeraTK::RegisterInfo::getTags,
            R"(Get the list of tags that are associated with this register.

        Returns:
            set[str]: Set of tags associated with this register.)");
  }
  /*******************************************************************************************************************/

  void RegisterInfo::bindBackendRegisterInfoBase(py::module& m) {
    py::class_<ChimeraTK::BackendRegisterInfoBase>(m, "BackendRegisterInfoBase")
        .def("getDataDescriptor", &ChimeraTK::BackendRegisterInfoBase::getDataDescriptor,
            R"(Return description of the actual payload data for this register.

        Returns:
            DataDescriptor: Object containing information about the data format.)")
        .def("isReadable", &ChimeraTK::BackendRegisterInfoBase::isReadable,
            R"(Return whether the register is readable.

        Returns:
            bool: True if the register is readable, false otherwise.)")
        .def("isWriteable", &ChimeraTK::BackendRegisterInfoBase::isWriteable,
            R"(Return whether the register is writeable.

        Returns:
            bool: True if the register is writeable, false otherwise.)")
        .def("getRegisterName", &ChimeraTK::BackendRegisterInfoBase::getRegisterName,
            R"(Return full path name of the register.

        Returns:
            RegisterPath: Full path name of the register (including modules).)")
        .def("getSupportedAccessModes", &ChimeraTK::BackendRegisterInfoBase::getSupportedAccessModes,
            R"(Return all supported AccessModes for this register.

        Returns:
            list[AccessMode]: Flags indicating supported access modes.)")
        .def("getNumberOfElements", &ChimeraTK::BackendRegisterInfoBase::getNumberOfElements,
            R"(Return number of elements per channel.

        Returns:
            int: Number of elements per channel.)")
        .def("getNumberOfDimensions", &ChimeraTK::BackendRegisterInfoBase::getNumberOfDimensions,
            R"(Return number of dimensions of this register.

        Returns:
            int: Number of dimensions (0=scalar, 1=1D array, 2=2D array).)")
        .def("getNumberOfChannels", &ChimeraTK::BackendRegisterInfoBase::getNumberOfChannels,
            R"(Return number of channels in register.

        Returns:
            int: Number of channels.)")
        .def("getQualifiedAsyncId", &ChimeraTK::BackendRegisterInfoBase::getQualifiedAsyncId,
            R"(Return the fully qualified async::SubDomain ID.

        The default implementation returns an empty vector.

        Returns:
            list[int]: List of IDs forming the fully qualified async::SubDomain ID.)")
        .def("getTags", &ChimeraTK::BackendRegisterInfoBase::getTags,
            R"(Get the list of tags associated with this register.

        The default implementation returns an empty set.

        Returns:
            set[str]: Set of tags associated with this register.)")
        .def("isHidden", &ChimeraTK::BackendRegisterInfoBase::isHidden,
            R"(Return whether the register is "hidden".

        Hidden registers won't be listed when iterating the catalogue, but can be explicitly iterated.

        Returns:
            bool: True if the register is hidden, false otherwise.)");
  }

  /*******************************************************************************************************************/

  ChimeraTK::DataDescriptor::FundamentalType DataDescriptor::fundamentalType(ChimeraTK::DataDescriptor& self) {
    return self.fundamentalType();
  }

  /*******************************************************************************************************************/

  void DataDescriptor::bind(py::module& m) {
    py::class_<ChimeraTK::DataDescriptor>(m, "DataDescriptor")
        .def(py::init<ChimeraTK::DataDescriptor>())
        .def(py::init<ChimeraTK::DataType>(), py::arg("type"),
            R"(Construct a DataDescriptor from a DataType object.

        The DataDescriptor will describe the passed DataType with no raw type.

        Args:
            type (DataType): The data type to describe.)")
        .def(py::init<>(),
            R"(Default constructor.

        Initializes the DataDescriptor with fundamental type set to "undefined".)")
        .def("fundamentalType", DeviceAccessPython::DataDescriptor::fundamentalType,
            R"(Get the fundamental data type.

        Returns:
            FundamentalType: The fundamental data type.)")
        .def("isSigned", &ChimeraTK::DataDescriptor::isSigned,
            R"(Return whether the data is signed or not.

        Only valid for numeric data types.

        Returns:
            bool: True if the data is signed, false otherwise.)")
        .def("isIntegral", &ChimeraTK::DataDescriptor::isIntegral,
            R"(Return whether the data is integral or not.

        May only be called for numeric data types. Examples: int or float.

        Returns:
            bool: True if the data is integral, false otherwise.)")
        .def("nDigits", &ChimeraTK::DataDescriptor::nDigits,
            R"(Return the approximate maximum number of digits needed to represent the value.

        This includes a decimal dot (if not an integral data type) and the sign.
        May only be called for numeric data types.

        Note: This number should only be used for displaying purposes. For some data types
        this might be a large number (e.g. 300), which indicates that a different representation
        than plain decimal numbers should be chosen.

        Returns:
            int: Approximate maximum number of digits (base 10).)")
        .def("nFractionalDigits", &ChimeraTK::DataDescriptor::nFractionalDigits,
            R"(Return the approximate maximum number of digits after the decimal dot.

        This is expressed in base 10 and excludes the decimal dot itself.
        May only be called for non-integral numeric data types.

        Note: This number should only be used for displaying purposes. There is no guarantee
        that the full precision can be displayed with the given number of digits.

        Returns:
            int: Approximate maximum number of fractional digits (base 10).)")
        .def("rawDataType", &ChimeraTK::DataDescriptor::rawDataType,
            R"(Get the raw data type.

        This describes the data conversion from 'cooked' to raw data type on the device.
        The conversion does not change the shape of the data but describes the data type of
        a single data point.

        Most backends will have type 'none' (no raw data conversion available).

        Returns:
            DataType: The raw data type.)")
        .def("setRawDataType", &ChimeraTK::DataDescriptor::setRawDataType, py::arg("rawDataType"),
            R"(Set the raw data type.

        This is useful e.g. when a decorated register should no longer allow raw access,
        in which case you should set DataType.none.

        Args:
            rawDataType (DataType): The raw data type to set.)")
        .def("transportLayerDataType", &ChimeraTK::DataDescriptor::transportLayerDataType,
            R"(Get the data type on the transport layer.

        This is always a 1D array of the specific data type. The raw transfer might contain
        data for more than one register.

        Examples:
            - The multiplexed data of a 2D array
            - A text string containing data for multiple scalars mapped to different registers
            - The byte sequence of a "struct" with data for multiple registers of different types

        Note: Currently all implementations return 'none'. There is no public API to access
        the transport layer data yet.

        Returns:
            DataType: The transport layer data type.)")
        .def("minimumDataType", &ChimeraTK::DataDescriptor::minimumDataType,
            R"(Get the minimum data type required to represent the described data type.

        This is the minimum data type needed in the host CPU to represent the value.

        Returns:
            DataType: The minimum required data type.)");
  }

  /*******************************************************************************************************************/

} /* namespace DeviceAccessPython*/
