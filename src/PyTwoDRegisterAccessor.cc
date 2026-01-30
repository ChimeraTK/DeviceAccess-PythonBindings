// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyTwoDRegisterAccessor.h"

#include <ChimeraTK/SupportedUserTypes.h>

#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/

  void PyTwoDRegisterAccessor::copyToBuffer() {
    std::visit(
        [&](auto& acc) {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using UserType = typename ACC::value_type;

          auto& buffer = std::get<std::vector<UserType>>(_continuousBuffer);
          buffer.resize(acc.getNChannels() * acc.getNElementsPerChannel());

          auto* out = buffer.data();
          for(size_t i = 0; i < acc.getNChannels(); ++i) {
            out = std::ranges::copy(acc[i], out).out;
          }
        },
        _accessor);
  }

  /********************************************************************************************************************/

  void PyTwoDRegisterAccessor::copyFromBuffer() {
    std::visit(
        [&](auto& acc) {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using UserType = typename ACC::value_type;

          auto& buffer = std::get<std::vector<UserType>>(_continuousBuffer);
          assert(buffer.size() == acc.getNChannels() * acc.getNElementsPerChannel());
          for(size_t i = 0; i < acc.getNChannels(); ++i) {
            auto in =
                std::span<UserType>(buffer.data() + i * acc.getNElementsPerChannel(), acc.getNElementsPerChannel());
            std::ranges::copy(in, acc[i].data());
          }
        },
        _accessor);
  }

  /********************************************************************************************************************/

  void PyTwoDRegisterAccessor::read() {
    py::gil_scoped_release release;
    visit([&](auto& acc) { acc.read(); });
    copyToBuffer();
  }

  /********************************************************************************************************************/

  bool PyTwoDRegisterAccessor::readLatest() {
    py::gil_scoped_release release;
    bool rv = visit([&](auto& acc) -> bool { return acc.readLatest(); });
    copyToBuffer();
    return rv;
  }

  /********************************************************************************************************************/

  bool PyTwoDRegisterAccessor::readNonBlocking() {
    py::gil_scoped_release release;
    bool rv = visit([&](auto& acc) -> bool { return acc.readNonBlocking(); });
    copyToBuffer();
    return rv;
  }

  /********************************************************************************************************************/

  void PyTwoDRegisterAccessor::write(const ChimeraTK::VersionNumber& versionNumber) {
    {
      py::gil_scoped_release release;
      copyFromBuffer();
    }
    PyTransferElement<PyTwoDRegisterAccessor>::write(versionNumber);
  }

  /********************************************************************************************************************/

  void PyTwoDRegisterAccessor::writeDestructively(const ChimeraTK::VersionNumber& versionNumber) {
    {
      py::gil_scoped_release release;
      copyFromBuffer();
    }
    PyTransferElement<PyTwoDRegisterAccessor>::writeDestructively(versionNumber);
  }

  /********************************************************************************************************************/

  size_t PyTwoDRegisterAccessor::getNElementsPerChannel() {
    size_t rv;
    std::visit([&](auto& acc) { rv = acc.getNElementsPerChannel(); }, _accessor);
    return rv;
  }

  /********************************************************************************************************************/

  size_t PyTwoDRegisterAccessor::getNChannels() {
    size_t rv;
    std::visit([&](auto& acc) { rv = acc.getNChannels(); }, _accessor);
    return rv;
  }

  /********************************************************************************************************************/

  UserTypeVariantNoVoid PyTwoDRegisterAccessor::getAsCooked(uint channel, uint element) {
    {
      py::gil_scoped_release release;
      copyFromBuffer();
    }
    return std::visit([&](auto& acc) { return acc.template getAsCooked<double>(channel, element); }, _accessor);
  }

  /********************************************************************************************************************/

  void PyTwoDRegisterAccessor::setAsCooked(uint channel, uint element, UserTypeVariantNoVoid value) {
    std::visit(
        [&](auto& acc) { std::visit([&](auto& val) { acc.setAsCooked(channel, element, val); }, value); }, _accessor);
    {
      py::gil_scoped_release release;
      copyToBuffer();
    }
  }

  /********************************************************************************************************************/

  void PyTwoDRegisterAccessor::set(const UserTypeTemplateVariantNoVoid<VVector>& vec) {
    std::visit(
        [&](auto& acc) {
          std::visit(
              [&](auto& incoming) {
                using ACC = typename std::remove_reference<decltype(acc)>::type;
                using UserType = typename ACC::value_type;

                using VecType = typename std::remove_reference<decltype(incoming)>::type;
                using VecValueType = typename VecType::value_type::value_type;

                auto& buffer = std::get<std::vector<UserType>>(_continuousBuffer);

                buffer.resize(acc.getNChannels() * acc.getNElementsPerChannel());
                if constexpr(std::is_same_v<UserType, VecValueType>) {
                  auto* out = buffer.data();
                  for(size_t i = 0; i < acc.getNChannels(); ++i) {
                    out = std::ranges::copy(incoming[i], out).out;
                  }
                }
                else {
                  for(size_t i = 0; i < acc.getNChannels(); ++i) {
                    auto out = std::span<UserType>(
                        buffer.data() + i * acc.getNElementsPerChannel(), acc.getNElementsPerChannel());

                    std::transform(incoming[i].cbegin(), incoming[i].cend(), out.begin(),
                        ChimeraTK::userTypeToUserType<UserType, VecValueType>);
                  }
                }
              },
              vec);
        },
        _accessor);
  }

  /********************************************************************************************************************/

  py::object PyTwoDRegisterAccessor::get() const {
    return std::visit(
        [&](auto& acc) -> py::object {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using userType = typename ACC::value_type;
          auto ndacc = boost::dynamic_pointer_cast<NDRegisterAccessor<userType>>(acc.getHighLevelImplElement());
          auto nChannels = acc.getNChannels();
          auto nElements = acc.getNElementsPerChannel();
          auto& buffer = std::get<std::vector<userType>>(_continuousBuffer);

          if constexpr(std::is_same<userType, std::string>::value) {
            // String arrays are not really supported by numpy, so we return a list instead
            return py::cast(ndacc->accessChannels());
          }
          else if constexpr(std::is_same<userType, ChimeraTK::Boolean>::value) {
            auto ary = py::array(py::dtype::of<bool>(), {int64_t(nChannels), int64_t(nElements)},
                {int64_t((sizeof(userType) * nElements)), int64_t(sizeof(userType))}, buffer.data(), py::cast(this));
            assert(!ary.owndata()); // numpy must not own our buffers
            return ary;
          }
          else {
            auto ary = py::array(py::dtype::of<userType>(), {int64_t(nChannels), int64_t(nElements)},
                {int64_t((sizeof(userType) * nElements)), int64_t(sizeof(userType))}, buffer.data(), py::cast(this));
            assert(!ary.owndata()); // numpy must not own our buffers
            return ary;
          }
        },
        _accessor);

    // return py::array(getBufferInfo());
  }

  /********************************************************************************************************************/

  std::string PyTwoDRegisterAccessor::repr(py::object& acc) const {
    std::string rep{"<PyTwoDRegisterAccessor(type="};
    rep.append(py::cast<py::object>(py::cast(&acc).attr("getValueType")()).attr("__repr__")().cast<std::string>());
    rep.append(", name=");
    rep.append(py::cast(&acc).attr("getName")().cast<std::string>());
    rep.append(", data=");
    rep.append(py::cast<py::object>(py::cast(&acc).attr("__str__")()).cast<std::string>());
    rep.append(", versionNumber=");
    rep.append(py::cast<py::object>(py::cast(&acc).attr("getVersionNumber")()).attr("__repr__")().cast<std::string>());
    rep.append(", dataValidity=");
    rep.append(py::cast<py::object>(py::cast(&acc).attr("dataValidity")()).attr("__repr__")().cast<std::string>());
    rep.append(")>");
    return rep;
  }

  /********************************************************************************************************************/

  py::buffer_info PyTwoDRegisterAccessor::getBufferInfo() const {
    py::buffer_info info;
    std::visit(
        [&](auto& acc) {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using userType = typename ACC::value_type;
          auto ndacc = boost::dynamic_pointer_cast<NDRegisterAccessor<userType>>(acc.getHighLevelImplElement());
          auto nChannels = acc.getNChannels();
          auto nElements = acc.getNElementsPerChannel();

          if constexpr(std::is_same<userType, ChimeraTK::Boolean>::value) {
            info.format = py::format_descriptor<bool>::format();
          }
          else if constexpr(std::is_same<userType, std::string>::value) {
            // TODO: something is breaking here
            // info.format = py::format_descriptor<std::string>::format();
            return;
          }
          else {
            info.format = py::format_descriptor<userType>::format();
          }
          auto& buffer = std::get<std::vector<userType>>(_continuousBuffer);
          info.ptr = buffer.data();
          info.itemsize = sizeof(userType);
          info.ndim = 2;
          info.shape = {int64_t(nChannels), int64_t(nElements)};
          info.strides = {int64_t((sizeof(userType) * nElements)), sizeof(userType)};
        },
        _accessor);
    return info;
  }

  /********************************************************************************************************************/

  py::object PyTwoDRegisterAccessor::getitem(size_t index) const {
    return std::visit(
        [&](auto& acc) -> py::object {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using userType = typename ACC::value_type;
          auto ndacc = boost::dynamic_pointer_cast<NDRegisterAccessor<userType>>(acc.getHighLevelImplElement());
          auto nElements = acc.getNElementsPerChannel();
          auto& buffer = std::get<std::vector<userType>>(_continuousBuffer);

          if constexpr(std::is_same<userType, std::string>::value) {
            // String arrays are not really supported by numpy, so we return a list instead
            return py::cast(ndacc->accessChannels());
          }
          else if constexpr(std::is_same<userType, ChimeraTK::Boolean>::value) {
            auto ary = py::array(py::dtype::of<bool>(), {int64_t(nElements)}, {int64_t(sizeof(userType))},
                &buffer[index * nElements], py::cast(this));
            assert(!ary.owndata()); // numpy must not own our buffers
            return ary;
          }
          else {
            auto ary = py::array(py::dtype::of<userType>(), {int64_t(nElements)}, {int64_t(sizeof(userType))},
                &buffer[index * nElements], py::cast(this));
            assert(!ary.owndata()); // numpy must not own our buffers
            return ary;
          }
        },
        _accessor);
  }

  /********************************************************************************************************************/

  PyTwoDRegisterAccessor::~PyTwoDRegisterAccessor() = default;

  /********************************************************************************************************************/

  void PyTwoDRegisterAccessor::bind(py::module& m) {
    py::class_<PyTwoDRegisterAccessor, PyTransferElementBase> arrayacc(m, "TwoDRegisterAccessor",
        R"(Accessor class to read and write 2D registers transparently by using the accessor like a two-dimensional array.

      The accessor exposes channels and per-channel elements. Conversion to and from the UserType is handled by a
      data converter matching the register description (if applicable).

      Note:
        Create instances via Device.getTwoDRegisterAccessor(). Transfers between the device and the internal buffer
        need to be triggered using read() and write() before reading from or after writing to the buffer.)",
        py::buffer_protocol());
    arrayacc.def(py::init<>())
        .def_buffer(&PyTwoDRegisterAccessor::getBufferInfo)
        .def("read", &PyTwoDRegisterAccessor::read,
            R"(Read the data from the device.

        If AccessMode.wait_for_new_data was set, this function will block until new data has arrived. Otherwise it
        still might block for a short time until the data transfer was complete.)")
        .def("readNonBlocking", &PyTwoDRegisterAccessor::readNonBlocking,
            R"(Read the next value, if available in the input buffer.

        If AccessMode.wait_for_new_data was set, this function returns immediately and the return value indicates
        if a new value was available (true) or not (false).

        If AccessMode.wait_for_new_data was not set, this function is identical to read(), which will still return
        quickly. Depending on the actual transfer implementation, the backend might need to transfer data to obtain
        the current value before returning. Also this function is not guaranteed to be lock free. The return value
        will be always true in this mode.

        :return: True if new data was available, false otherwise.
        :rtype: bool)")
        .def("readLatest", &PyTwoDRegisterAccessor::readLatest,
            R"(Read the latest value, discarding any other update since the last read if present.

        Otherwise this function is identical to readNonBlocking(), i.e. it will never wait for new values and it
        will return whether a new value was available if AccessMode.wait_for_new_data is set.

        :return: True if new data was available, false otherwise.
        :rtype: bool)")
        .def("write", &PyTwoDRegisterAccessor::write,
            R"(Write the buffered data to the device.

        The return value is true if old data was lost on the write transfer (e.g. due to a buffer overflow).
        In case of an unbuffered write transfer, the return value will always be false.

        :param versionNumber: Version number to use for this write operation. If not specified, a new version
          number is generated.
        :type versionNumber: VersionNumber)",
            py::arg("versionNumber") = PyVersionNumber::getNullVersion())
        .def("writeDestructively", &PyTwoDRegisterAccessor::writeDestructively,
            R"(Just like write(), but allows the implementation to destroy the content of the user buffer in the
        process.

        This is an optional optimisation, hence there is a default implementation which just calls write(). In any
        case, the application must expect the user buffer of the accessor to contain undefined data after calling
        this function.

        :param versionNumber: Version number to use for this write operation. If not specified, a new version
          number is generated.
        :type versionNumber: VersionNumber)",
            py::arg("versionNumber") = PyVersionNumber::getNullVersion())
        .def("interrupt", &PyTwoDRegisterAccessor::interrupt,
            R"(Interrupt a blocking read operation.

        This will cause a blocking read to return immediately and throw an InterruptedException.)")
        .def("getName", &PyTwoDRegisterAccessor::getName,
            R"(Returns the name that identifies the process variable.

        :return: The register name.
        :rtype: str)")
        .def("getUnit", &PyTwoDRegisterAccessor::getUnit,
            R"(Returns the engineering unit.

        If none was specified, it will default to 'n./a.'.

        :return: The engineering unit string.
        :rtype: str)")
        .def("getDescription", &PyTwoDRegisterAccessor::getDescription,
            R"(Returns the description of this variable/register.

        :return: The description string.
        :rtype: str)")
        .def("getValueType", &PyTwoDRegisterAccessor::getValueType,
            R"(Returns the type_info for the value type of this accessor.

        This can be used to determine the type at runtime.

        :return: Type information object.
        :rtype: type)")
        .def("getVersionNumber", &PyTwoDRegisterAccessor::getVersionNumber,
            R"(Returns the version number that is associated with the last transfer.

        This refers to the last read or write operation.

        :return: The version number of the last transfer.
        :rtype: VersionNumber)")
        .def("isReadOnly", &PyTwoDRegisterAccessor::isReadOnly,
            R"(Check if accessor is read only.

        This means it is readable but not writeable.

        :return: True if read only, false otherwise.
        :rtype: bool)")
        .def("isReadable", &PyTwoDRegisterAccessor::isReadable,
            R"(Check if accessor is readable.

        :return: True if readable, false otherwise.
        :rtype: bool)")
        .def("isWriteable", &PyTwoDRegisterAccessor::isWriteable,
            R"(Check if accessor is writeable.

        :return: True if writeable, false otherwise.
        :rtype: bool)")
        .def("getId", &PyTwoDRegisterAccessor::getId,
            R"(Obtain unique ID for the actual implementation of this accessor.

        This means that e.g. two instances of TwoDRegisterAccessor created by the same call to
        Device.getTwoDRegisterAccessor() will have the same ID, while two instances obtained by two different
        calls to Device.getTwoDRegisterAccessor() will have a different ID even when accessing the very same
        register.

        :return: The unique transfer element ID.
        :rtype: TransferElementID)")
        .def("dataValidity", &PyTwoDRegisterAccessor::dataValidity,
            R"(Return current validity of the data.

        Will always return DataValidity.ok if the backend does not support it.

        :return: The current data validity state.
        :rtype: DataValidity)")
        .def("isInitialised", &PyTwoDRegisterAccessor::isInitialised,
            R"(Check if the accessor is initialised.

        :return: True if initialised, false otherwise.
        :rtype: bool)")
        .def("getNElementsPerChannel", &PyTwoDRegisterAccessor::getNElementsPerChannel,
            R"(Return number of elements/samples per channel in the register.

        :return: Number of elements per channel.
        :rtype: int)")
        .def("getNChannels", &PyTwoDRegisterAccessor::getNChannels,
            R"(Return number of channels in the register.

        :return: Number of channels.
        :rtype: int)")
        .def("get", &PyTwoDRegisterAccessor::get,
            R"(Return a 2D array of UserType from the internal buffer (without a previous read).

        The returned object is typically a numpy ndarray with shape (channels, elements_per_channel). For string
        registers, a list of lists is returned instead.

        :return: Current buffer content as a 2D array-like object.
        :rtype: ndarray or list)")
        .def("set", &PyTwoDRegisterAccessor::set, py::arg("newValue"),
            R"(Set the values of the 2D array buffer.

        :param newValue: New values to set, shaped as [channels][elements] or a 2D numpy array.
        :type newValue: list[list[UserType]] or ndarray)")
        .def("getAsCooked", &PyTwoDRegisterAccessor::getAsCooked,
            R"(Get a cooked value for a specific channel and element when the accessor is raw (no data conversion).

        This returns the converted data from the user buffer. It does not do any read or write transfer.

        :param channel: Channel index.
        :type channel: int
        :param element: Element index within the channel.
        :type element: int
        :return: The cooked value.
        :rtype: float)",
            py::arg("channel"), py::arg("element"))
        .def("setAsCooked", &PyTwoDRegisterAccessor::setAsCooked,
            R"(Set a cooked value for a specific channel and element when the accessor is raw (no data conversion).

        This converts to raw and writes the data to the user buffer. It does not do any read or write transfer.

        :param channel: Channel index.
        :type channel: int
        :param element: Element index within the channel.
        :type element: int
        :param value: The cooked value to set.
        :type value: float)",
            py::arg("channel"), py::arg("element"), py::arg("value"))
        .def("setDataValidity", &PyTwoDRegisterAccessor::setDataValidity,
            R"(Set the data validity of the accessor.

        :param validity: The data validity state to set.
        :type validity: DataValidity)")
        .def("getAccessModeFlags", &PyTwoDRegisterAccessor::getAccessModeFlags,
            R"(Return the access mode flags that were used to create this accessor.

        This can be used to determine the setting of the raw and the wait_for_new_data flags.

        :return: List of access mode flags.
        :rtype: list[AccessMode])")
        .def(
            "__getitem__", [](PyTwoDRegisterAccessor& acc, size_t index) { return acc.getitem(index); },
            py::arg("index"),
            R"(Get an element from the array by index.

            :param index: The element index.
            :type index: int
            :return: The value at the specified index.
            :rtype: scalar)")
        .def(
            "__getitem__",
            [](PyTwoDRegisterAccessor& acc, const py::object& slice) { return acc.get().attr("__getitem__")(slice); },
            py::arg("slice"),
            R"(Get an element from the array by index.

            :param slice: A slice object.
            :type slice: slice
            :return: The value(s) at the specified slice.
            :rtype: np.ndarray)")
        .def(
            "__setitem__",
            [](PyTwoDRegisterAccessor& acc, size_t index, const UserTypeVariantNoVoid& value) {
              acc.get().attr("__setitem__")(index, py::cast(value));
            },
            py::arg("index"), py::arg("value"),
            R"(Set an element in the array by index.

            :param index: The element index.
            :type index: int
            :param value: The value to set at the specified index.
            :type value: user type)")
        .def(
            "__setitem__",
            [](PyTwoDRegisterAccessor& acc, const py::object& slice, const UserTypeVariantNoVoid& value) {
              acc.get().attr("__setitem__")(slice, py::cast(value));
            },
            py::arg("slice"), py::arg("value"),
            R"(Set an element in the array by slice.

            :param slice: The element slice.
            :type slice: slice
            :param value: The value to set at the specified slice.
            :type value: user type)")
        .def(
            "__setitem__",
            [](PyTwoDRegisterAccessor& acc, const py::object& slice, const py::object& array) {
              acc.get().attr("__setitem__")(slice, array);
            },
            py::arg("slice"), py::arg("array"),
            R"(Set an element in the array by slice.

            :param slice: The element slice.
            :type slice: slice
            :param array: The value to set at the specified slice.
            :type array: list or ndarray)")
        .def("__getitem__", &PyTwoDRegisterAccessor::getitem, py::arg("index"),
            R"(Get a single channel by index.

        :param index: Channel index.
        :type index: int
        :return: View of the selected channel as a 1D array-like object.
        :rtype: ndarray or list)")
        .def("__getattr__", &PyTwoDRegisterAccessor::getattr);

    for(const auto& fn : PyTransferElementBase::specialFunctionsToEmulateNumeric) {
      arrayacc.def(fn.c_str(), [fn](PyTwoDRegisterAccessor& acc, PyTwoDRegisterAccessor& other) {
        return acc.get().attr(fn.c_str())(other.get());
      });
      arrayacc.def(fn.c_str(),
          [fn](PyTwoDRegisterAccessor& acc, py::object& other) { return acc.get().attr(fn.c_str())(other); });
    }

    for(const auto& fn : PyTransferElementBase::specialAssignmentFunctionsToEmulateNumeric) {
      arrayacc.def(
          fn.c_str(), [fn](PyTwoDRegisterAccessor& acc, PyTwoDRegisterAccessor& other) -> PyTwoDRegisterAccessor& {
            acc.get().attr(fn.c_str())(other.get());
            return acc;
          });

      arrayacc.def(fn.c_str(), [fn](PyTwoDRegisterAccessor& acc, py::object& other) -> PyTwoDRegisterAccessor& {
        acc.get().attr(fn.c_str())(other);
        return acc;
      });
    }

    for(const auto& fn : PyTransferElementBase::specialUnaryFunctionsToEmulateNumeric) {
      arrayacc.def(fn.c_str(), [fn](PyTwoDRegisterAccessor& acc) { return acc.get().attr(fn.c_str())(); });
    }
  }

  /********************************************************************************************************************/

} // namespace ChimeraTK
