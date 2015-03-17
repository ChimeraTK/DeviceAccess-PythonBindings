#include <MtcaMappedDevice/devPCIE.h>
#include <MtcaMappedDevice/DummyDevice.h>
#include "PythonInterfaceWrapper.h"
#include "devBaseFactory.h"
#include "devMapFactory.h"
#include "MtcaMappedDevice/exDevPCIE.h"
#include <numpy/arrayobject.h>

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

// This section defines function pointers used for overloading methods//
//**********************************************************************************//
static void (mtca4upy::PythonInterface::*readDMAUsingRegisterOffset)(
    uint32_t, bp::numeric::array, size_t) = &mtca4upy::PythonInterface::readDMA;
static void (mtca4upy::PythonInterface::*writeDMAUsingRegisterOffset)(
    uint32_t, bp::numeric::array, size_t) =
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

static void (mtca4upy::PythonInterface::*readRawUsingRegisterOffset)(
    uint32_t, bp::numeric::array, size_t, uint8_t) =
    &mtca4upy::PythonInterface::readRaw;

static void (mtca4upy::PythonInterface::*readRawUsingRegisterName)(
    const std::string&, bp::numeric::array, size_t, uint32_t) =
    &mtca4upy::PythonInterface::readRaw;

static void (mtca4upy::PythonInterface::*writeRawUsingRegisterOffset)(
    uint32_t, bp::numeric::array, size_t, uint8_t) =
    &mtca4upy::PythonInterface::writeRaw;
static void (mtca4upy::PythonInterface::*writeRawUsingRegisterName)(
    const std::string&, bp::numeric::array, size_t, uint32_t) =
    &mtca4upy::PythonInterface::writeRaw;

//**********************************************************************************//

//wrapper methods for read and write
float* extractDataPointer(const bp::numeric::array& Buffer) {
  PyArrayObject* pointerToNumPyArrayMemory =
      reinterpret_cast<PyArrayObject*>(Buffer.ptr());
  return (reinterpret_cast<float*>(pointerToNumPyArrayMemory->data));
}

void readWrapper(mtca4u::devMap<mtca4u::devBase>::RegisterAccessor &self, bp::numeric::array& dataSpace,size_t arraySize, uint32_t elementIndexInRegister){
  float* dataLocation = extractDataPointer(dataSpace);
  uint32_t dataOffset = elementIndexInRegister * sizeof(uint32_t); // This is assuming that the PCIE mapped memory increments as a 32 bit word for each element
  self.read<float>(dataLocation, arraySize, dataOffset);
}

void writeWrapper(mtca4u::devMap<mtca4u::devBase>::RegisterAccessor &self, bp::numeric::array& dataSpace,size_t arraySize, uint32_t dataOffset){
  float* dataLocation = extractDataPointer(dataSpace);
  self.write<float>(dataLocation, arraySize, dataOffset);
}

uint32_t sizeWrapper(mtca4u::devMap<mtca4u::devBase>::RegisterAccessor &self){
  mtca4u::mapFile::mapElem mapelem = self.getRegisterInfo();
  return(mapelem.reg_elem_nr);
}

BOOST_PYTHON_MODULE(mtcamappeddevice) {
  bp::class_<mtca4upy::PythonInterfaceWrapper, boost::noncopyable>("Device")
      .def("readRaw", bp::pure_virtual(readRawUsingRegisterOffset))
      .def("readRaw", bp::pure_virtual(readRawUsingRegisterName))
      .def("writeRaw", bp::pure_virtual(writeRawUsingRegisterOffset))
      .def("writeRaw", bp::pure_virtual(writeRawUsingRegisterName))
      .def("readDMA", bp::pure_virtual(readDMAUsingRegisterOffset))
      .def("readDMA", bp::pure_virtual(readDMAUsingRegisterName))
      .def("writeDMA", bp::pure_virtual(writeDMAUsingRegisterOffset))
      .def("getRegisterAccessor", bp::pure_virtual(&mtca4upy::PythonInterface::getRegisterAccessor));

  bp::class_<mtca4u::devMap<mtca4u::devBase>::RegisterAccessor>(
      "RegisterAccessor",
      bp::init<const std::string, const mtca4u::mapFile::mapElem,
               boost::shared_ptr<mtca4u::devBase> >())
        	   .def("read", readWrapper)
		   .def("write", writeWrapper)
		   .def("getNumElements", sizeWrapper);

  bp::def("createDevice", createDevice);
  bp::def("createDevice", createMappedDevice);
  bp::register_ptr_to_python<boost::shared_ptr<mtca4upy::PythonInterface> >();
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
}
