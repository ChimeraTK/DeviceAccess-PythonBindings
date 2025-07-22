// SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "PyTwoDRegisterAccessor.h"

#include <ChimeraTK/SupportedUserTypes.h>

#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <vector>

namespace py = pybind11;

namespace ChimeraTK {

  /********************************************************************************************************************/
  void PyTwoDRegisterAccessor::copyToBuffer() {
    std::visit(
        [&](auto& acc) {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using UserType = typename ACC::value_type;

          _continuousBuffer = std::vector<UserType>{};
          auto& buffer = std::get<std::vector<UserType>>(_continuousBuffer);
          buffer.clear();
          buffer.reserve(acc.getNChannels() * acc.getNElementsPerChannel());
          for(size_t i = 0; i < acc.getNChannels(); ++i) {
            buffer.insert(buffer.end(), acc[i].begin(), acc[i].end());
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

  void PyTwoDRegisterAccessor::readLatest() {
    py::gil_scoped_release release;
    visit([&](auto& acc) { acc.readLatest(); });
    copyToBuffer();
  }

  /********************************************************************************************************************/

  void PyTwoDRegisterAccessor::readNonBlocking() {
    py::gil_scoped_release release;
    visit([&](auto& acc) { acc.readNonBlocking(); });
    copyToBuffer();
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

  void PyTwoDRegisterAccessor::set(const UserTypeTemplateVariantNoVoid<VVector>& vec) {
    std::visit(
        [&](auto& acc) {
          using ACC = typename std::remove_reference<decltype(acc)>::type;
          using expectedUserType = typename ACC::value_type;
          std::visit(
              [&](const auto& outerVector) {
                // Resize acc to match the input shape
                for(size_t i = 0; i < outerVector.size(); ++i) {
                  for(size_t j = 0; j < outerVector[i].size(); ++j) {
                    acc[i][j] = userTypeToUserType<expectedUserType>(outerVector[i][j]);
                  }
                }
              },
              vec);
        },
        _accessor);
  }
  /********************************************************************************************************************/

  py::object PyTwoDRegisterAccessor::get() const {
    return py::array(getBufferInfo());
  }
  /********************************************************************************************************************/

  std::string PyTwoDRegisterAccessor::repr(py::object& acc) const {
    std::string rep{"<TwoDRegisterAccessor(type="};
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
    py::object rv;
    std::visit([&](auto& acc) { rv = py::cast(acc[index]); }, _accessor);
    return rv;
  }

  /********************************************************************************************************************/

  PyTwoDRegisterAccessor::~PyTwoDRegisterAccessor() = default;

  /********************************************************************************************************************/

  void PyTwoDRegisterAccessor::bind(py::module& m) {
    py::class_<PyTwoDRegisterAccessor, PyTransferElementBase> arrayacc(
        m, "TwoDRegisterAccessor", py::buffer_protocol());
    arrayacc.def(py::init<>())
        .def_buffer(&PyTwoDRegisterAccessor::getBufferInfo)
        .def("read", &PyTwoDRegisterAccessor::read,
            "Read the data from the device.\n\nIf AccessMode::wait_for_new_data was set, this function will block "
            "until new data has arrived. Otherwise it still might block for a short time until the data transfer was "
            "complete.")
        .def("readNonBlocking", &PyTwoDRegisterAccessor::readNonBlocking,
            "Read the next value, if available in the input buffer.\n\nIf AccessMode::wait_for_new_data was set, "
            "this "
            "function returns immediately and the return value indicated if a new value was available (true) or not "
            "(false).\n\nIf AccessMode::wait_for_new_data was not set, this function is identical to read(), which "
            "will still return quickly. Depending on the actual transfer implementation, the backend might need to "
            "transfer data to obtain the current value before returning. Also this function is not guaranteed to be "
            "lock free. The return value will be always true in this mode.")
        .def("readLatest", &PyTwoDRegisterAccessor::readLatest,
            "Read the latest value, discarding any other update since the last read if present.\n\nOtherwise this "
            "function is identical to readNonBlocking(), i.e. it will never wait for new values and it will return "
            "whether a new value was available if AccessMode::wait_for_new_data is set.")
        .def("write", &PyTwoDRegisterAccessor::write,
            "Write the data to device.\n\nThe return value is true, old data was lost on the write transfer (e.g. "
            "due "
            "to an buffer overflow). In case of an unbuffered write transfer, the return value will always be false.",
            py::arg("versionNumber") = PyVersionNumber::getNullVersion())
        .def("writeDestructively", &PyTwoDRegisterAccessor::writeDestructively,
            "Just like write(), but allows the implementation to destroy the content of the user buffer in the "
            "process.\n\nThis is an optional optimisation, hence there is a default implementation which just calls "
            "the normal doWriteTransfer(). In any case, the application must expect the user buffer of the "
            "TransferElement to contain undefined data after calling this function.",
            py::arg("versionNumber") = PyVersionNumber::getNullVersion())
        .def("getName", &PyTwoDRegisterAccessor::getName, "Returns the name that identifies the process variable.")
        .def("getUnit", &PyTwoDRegisterAccessor::getUnit,
            "Returns the engineering unit.\n\nIf none was specified, it will default to ' n./ a.'")
        .def("getDescription", &PyTwoDRegisterAccessor::getDescription,
            "Returns the description of this variable/register.")
        .def("getValueType", &PyTwoDRegisterAccessor::getValueType,
            "Returns the std::type_info for the value type of this transfer element.\n\nThis can be used to "
            "determine "
            "the type at runtime.")
        .def("getVersionNumber", &PyTwoDRegisterAccessor::getVersionNumber,
            "Returns the version number that is associated with the last transfer (i.e. last read or write)")
        .def("isReadOnly", &PyTwoDRegisterAccessor::isReadOnly,
            "Check if transfer element is read only, i.e. it is readable but not writeable.")
        .def("isReadable", &PyTwoDRegisterAccessor::isReadable, "Check if transfer element is readable.")
        .def("isWriteable", &PyTwoDRegisterAccessor::isWriteable, "Check if transfer element is writeable.")
        .def("getId", &PyTwoDRegisterAccessor::getId,
            "Obtain unique ID for the actual implementation of this TransferElement.\n\nThis means that e.g. two "
            "instances of ScalarRegisterAccessor created by the same call to Device::getScalarRegisterAccessor() "
            "(e.g. "
            "by copying the accessor to another using NDRegisterAccessorBridge::replace()) will have the same ID, "
            "while two instances obtained by to difference calls to Device::getScalarRegisterAccessor() will have a "
            "different ID even when accessing the very same register.")
        .def("dataValidity", &PyTwoDRegisterAccessor::dataValidity,
            "Return current validity of the data.\n\nWill always return DataValidity.ok if the backend does not "
            "support it")
        .def("isInitialised", &PyTwoDRegisterAccessor::isInitialised, "Check if the transfer element is initialised.")
        .def("getNElementsPerChannel", &PyTwoDRegisterAccessor::getNElementsPerChannel,
            "Return number of elements/samples per Channel in the register.")
        .def("getNChannels", &PyTwoDRegisterAccessor::getNChannels, "Return number of Channels in the register.")
        .def("get", &PyTwoDRegisterAccessor::get, "Return an array of UserType (without a previous read).")
        .def("set", &PyTwoDRegisterAccessor::set, "Set the values of the array of UserType.", py::arg("newValue"))
        .def("setDataValidity", &PyTwoDRegisterAccessor::setDataValidity,
            "Set the data validity of the transfer element.")
        .def("getAccessModeFlags", &PyTwoDRegisterAccessor::getAccessModeFlags,
            "Return the access mode flags that were used to create this TransferElement.\n\nThis can be used to "
            "determine the setting of the `raw` and the `wait_for_new_data` flags")
        .def("__getitem__", &PyTwoDRegisterAccessor::getitem, "Get an element from the array by index.")
        .def("__getattr__", &PyTwoDRegisterAccessor::getattr);
    for(const auto& fn : PyTransferElementBase::specialFunctionsToEmulateNumeric) {
      arrayacc.def(fn.c_str(), [fn](PyTwoDRegisterAccessor& acc, PyTwoDRegisterAccessor& other) {
        return acc.get().attr(fn.c_str())(other.get());
      });
      arrayacc.def(fn.c_str(),
          [fn](PyTwoDRegisterAccessor& acc, py::object& other) { return acc.get().attr(fn.c_str())(other); });
    }
    for(const auto& fn : PyTransferElementBase::specialUnaryFunctionsToEmulateNumeric) {
      arrayacc.def(fn.c_str(), [fn](PyTwoDRegisterAccessor& acc) { return acc.get().attr(fn.c_str())(); });
    }
  }

  /********************************************************************************************************************/

} // namespace ChimeraTK