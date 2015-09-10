#include <MtcaMappedDevice/devPCIE.h>
#include <MtcaMappedDevice/DummyDevice.h>
#include "devMapFactory.h"
#include "devBaseFactory.h"
#include "PythonDeviceWrapper.h"
#include "PythonDeviceWrapper.h"
#include "RegisterAccessorWrapperFunctions.h"
#include "MultiplexedDataAccessorWrapper.h"

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
  mtca4upy::devBaseFactory baseDevice(
      new mtca4upy::DeviceInformation(cardName, ""));
  return (baseDevice.createDevice());
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
           bp::pure_virtual(&mtca4upy::PythonDevice::getRegisterAccessor))
      .def("getMultiplexedDataAccessor",
           bp::pure_virtual(&mtca4upy::PythonDevice::getMultiplexedDataAccessor));

  bp::class_<
      mtca4u::devMap<mtca4u::devBase>::RegisterAccessor,
      boost::shared_ptr<mtca4u::devMap<mtca4u::devBase>::RegisterAccessor> >(
      "RegisterAccessor",
      bp::init<const std::string, const mtca4u::mapFile::mapElem,
               boost::shared_ptr<mtca4u::devBase> >())
      .def("read", mtca4upy::RegisterAccessor::readWrapper)
      .def("write", mtca4upy::RegisterAccessor::writeWrapper)
      .def("readRaw", mtca4upy::RegisterAccessor::readRawWrapper)
      .def("writeRaw", mtca4upy::RegisterAccessor::writeRawWrapper)
      .def("readDMARaw", mtca4upy::RegisterAccessor::readDMARawWrapper)
      .def("getNumElements", mtca4upy::RegisterAccessor::sizeWrapper);

     /*
     * define the multipleddataaccessor class in the module. here we specify its
     * python interface in the boost generated python module.
     *
     * The MultiplexedDataAccessor is an abstract class. We would also need to
     * register a shared pointer object to this type (because the factory we use
     * to create it gives back a shared pointer). What we have below should take
     * care of the abstract class part
     */
  bp::class_<mtca4upy::MultiplexedDataAccessorWrapper, boost::noncopyable>(
      "MuxDataAccessor",
      bp::init<const boost::shared_ptr<mtca4u::devBase>&,
               const std::vector<mtca4u::FixedPointConverter>&>())
      .def("readFromDevice", mtca4upy::MuxDataAccessor::readInDataFromCard)
      .def("getSequenceCount", mtca4upy::MuxDataAccessor::getSequenceCount)
      .def("getBlockCount", mtca4upy::MuxDataAccessor::getBlockCount)
      .def("populateArray", mtca4upy::MuxDataAccessor::copyReadInData);


  bp::def("createDevice", createDevice);
  bp::def("createDevice", createMappedDevice);
  bp::register_ptr_to_python<boost::shared_ptr<mtca4upy::PythonDevice> >();
  bp::register_ptr_to_python<
      boost::shared_ptr<mtca4u::MultiplexedDataAccessor<float> > >();
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
}
