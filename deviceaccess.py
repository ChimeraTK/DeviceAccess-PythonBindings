import mtca4udeviceaccess
import numpy as np
import enum


def setDMapFilePath(dmapFilePath):
    # dmapFilePath	Relative or absolute path of the dmap file (directory and file name).
    mtca4udeviceaccess.setDmapFile(dmapFilePath)


def getDMapFilePath(dmapFilePath):
    return mtca4udeviceaccess.getDmapFile()


class Device:
    def __init__(self, aliasName=None):
        if aliasName:
            self.aliasName = aliasName
            self._device = mtca4udeviceaccess.getDevice(aliasName)
        else:
            self._device = mtca4udeviceaccess.getDevice()

    def open(self, aliasName=None):
        if not aliasName:
            if self.aliasName:
                self._device.open()
            else:
                raise SyntaxError(
                    "No backend is assigned: the device is not opened"
                )
        elif aliasName is self.aliasName:
            self._device.open(aliasName)
        else:
            raise SyntaxError(
                "Device has not been opened correctly: the device is not opened"
            )

    def close(self):
        self._device.close()

    def getTwoDRegisterAccessor(self, userType, registerPathName, numberOfElements=0, elementsOffset=0, AccessModeFlags=None):
        if userType is np.int32:
            accessor = self._device.getTwoDAccessor_int32(
                registerPathName, numberOfElements, elementsOffset)
        else:
            raise SyntaxError(
                "userType not supported"
            )

        return accessor.getBuffer()


class AccessMode(enum.Enum):
    raw = mtca4udeviceaccess.AccessMode.raw
    wait_for_new_data = mtca4udeviceaccess.AccessMode.wait_for_new_data
