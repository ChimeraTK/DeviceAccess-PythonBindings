#include <MtcaMappedDevice/devPCIE.h>
#include <MtcaMappedDevice/DummyDevice.h>
#include "devMapFactory.h"
#include "devBaseFactory.h"
#include "PythonDeviceWrapper.h"
#include "PythonDeviceWrapper.h"
#include "RegisterAccessorWrapperFunctions.h"

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

boost::shared_ptr<mtca4upy::PythonDevice> createDevice(
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
boost::shared_ptr<mtca4upy::PythonDevice> createDevice(
    const std::string& cardName, const std::string& mapFile) {
  mtca4upy::devMapFactory mappedDeviceFactory(
      new mtca4upy::DeviceInformation(cardName, mapFile));
  return (mappedDeviceFactory.createDevice());
}
} /* namespace mtca4upy */

// This section defines function pointers used for overloading methods//
//****************************************************************************//

static boost::shared_ptr<mtca4upy::PythonDevice>(*createDevice)(
    const std::string&) = &mtca4upy::createDevice;

static boost::shared_ptr<mtca4upy::PythonDevice>(*createMappedDevice)(
    const std::string&, const std::string&) = &mtca4upy::createDevice;

//****************************************************************************//

BOOST_PYTHON_MODULE(mtcamappeddevice) { // TODO: find a better name for
                                        // mtcamappeddevice. This module is
                                        // actally accessed through mtca4upy.py
                                        // and not directly
  bp::class_<mtca4upy::PythonDeviceWrapper, boost::noncopyable>(
      "Device") // TODO: Find and change "Device" to a suitable name
      .def("writeRaw", bp::pure_virtual(&mtca4upy::PythonDevice::writeRaw))
      .def("getRegisterAccessor",
           bp::pure_virtual(&mtca4upy::PythonDevice::getRegisterAccessor));

  bp::class_<mtca4u::devMap<mtca4u::devBase>::RegisterAccessor>(
      "RegisterAccessor",
      bp::init<const std::string, const mtca4u::mapFile::mapElem,
               boost::shared_ptr<mtca4u::devBase> >())
      .def("read", mtca4upy::readWrapper)
      .def("write", mtca4upy::writeWrapper)
      .def("readRaw", mtca4upy::readRawWrapper)
      .def("writeRaw", mtca4upy::writeRawWrapper)
      .def("readDMARaw", mtca4upy::readDMARawWrapper)
      .def("getNumElements", mtca4upy::sizeWrapper);

  bp::def("createDevice", createDevice);
  bp::def("createDevice", createMappedDevice);
  bp::register_ptr_to_python<boost::shared_ptr<mtca4upy::PythonDevice> >();
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
}
