#include <MtcaMappedDevice/devPCIE.h>
#include "PythonInterfaceWrapper.h"
#include "devBaseFactory.h"
#include "devMapFactory.h"
#include "MtcaMappedDevice/exDevPCIE.h"

namespace mtca4upy {
/**
 * Currently 4 variants of devices are supported. These are:
 * - The PCIE device (mtca4u::devPCIE)
 * - The Dummy device (mtca4u::DummyDevice)
 * - The Mapped PCIE device (mtca4u::devMap<mtca4u::devPCIE>)
 * - The Mapped Dummy device (mtca4u::devMap<mtca4u::DummyDevice>)
 *
 * Depending on the call to create device, either of these 4 devices are
 *returned.
 */

/**
 * createDevice called with only the cardName parameter returns either a devPCIE
 * device or a DummyDevice. An opened devPCIE device is returned if the supplied
 * cardName corresponds to the device name. An opened DummyDevice is returned if
 * the cardName is a Mapfile.
 */

boost::shared_ptr<mtca4upy::PythonInterface> createDevice(
    const std::string& cardName) {
  mtca4upy::devBaseFactory deviceFactory(
      new mtca4upy::DeviceInformation(cardName, ""));
  return (deviceFactory.createDevice());
}

/**
 * This version of createDevice returns either an opened mapped PCIE device or a
 * Mapped Dummy device. A mapped PCIE device is returned if the cardName
 * corresponds to the device name and the mapFile parameter corresponds to a
 * valid map file. A mapped Dummy device is returned if both the  cardName and
 * mapFile parameters are set to the same valid Map file.
 */
boost::shared_ptr<mtca4upy::PythonInterface> createDevice(
    const std::string& cardName, const std::string& mapFile) {
  mtca4upy::devMapFactory mappedDeviceFactory(
      new mtca4upy::DeviceInformation(cardName, mapFile));
  return (mappedDeviceFactory.createDevice());
}
} /* namespace mtca4upy */

void translate(mtca4u::exDevPCIE const& e) {
  PyErr_SetString(PyExc_RuntimeError, e.what());
}

// This section defines function pointers used for overloading methods//
//**********************************************************************************//
static void (mtca4upy::PythonInterface::*readDMAUsingRegisterOffset)(
    uint32_t, bp::numeric::array, size_t, uint8_t) =
    &mtca4upy::PythonInterface::readDMA;
static void (mtca4upy::PythonInterface::*writeDMAUsingRegisterOffset)(
    uint32_t, bp::numeric::array, size_t, uint8_t) =
    &mtca4upy::PythonInterface::writeDMA;
static void (mtca4upy::PythonInterface::*readDMAUsingRegisterName)(
    const std::string&, bp::numeric::array, size_t, uint32_t) =
    &mtca4upy::PythonInterface::readDMA;
static void (mtca4upy::PythonInterface::*writeDMAUsingRegisterName)(
    const std::string&, bp::numeric::array, size_t, uint32_t) =
    &mtca4upy::PythonInterface::writeDMA;
static boost::shared_ptr<mtca4upy::PythonInterface>(*createDevice)(
    const std::string&) = &mtca4upy::createDevice;
static boost::shared_ptr<mtca4upy::PythonInterface>(*createMappedDevice)(
    const std::string&, const std::string&) = &mtca4upy::createDevice;
//**********************************************************************************//

BOOST_PYTHON_MODULE(mtcamappeddevice) {
  bp::register_exception_translator<mtca4u::exDevPCIE>(&translate);

  bp::class_<mtca4upy::PythonInterfaceWrapper, boost::noncopyable>("Device")
      .def("readArea", bp::pure_virtual(&mtca4upy::PythonInterface::readArea))
      .def("writeArea", bp::pure_virtual(&mtca4upy::PythonInterface::writeArea))
      .def("readDMA", bp::pure_virtual(readDMAUsingRegisterOffset))
      .def("readDMA", bp::pure_virtual(readDMAUsingRegisterName))
      .def("writeDMA", bp::pure_virtual(writeDMAUsingRegisterOffset))
      .def("writeDMA", bp::pure_virtual(writeDMAUsingRegisterName));

  bp::def("createDevice", createDevice);
  bp::def("createDevice", createMappedDevice);
  bp::register_ptr_to_python<boost::shared_ptr<mtca4upy::PythonInterface> >();
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
}
