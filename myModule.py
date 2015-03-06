import mtcamappeddevice 
import numpy


class Device():

    def __init__(self, deviceName, mapFile = None):
        # exception handling is a kludge for now
        try:
            mtcamappeddevice.createDevice(5)
        except Exception as self.__storedArgErrException:
            pass

        try:
            if (mapFile != None):
                self.__openedDevice = mtcamappeddevice.createDevice(deviceName,
                        mapFile)
            else:
                self.__openedDevice = mtcamappeddevice.createDevice(deviceName)
        except Exception, e:            
            if(e.__class__ == self.__storedArgErrException.__class__):
                print "Device name and mapfile name are expected to be strings"

    def readRaw(self, offset, numberOfWords, registerName=None, bar=0):

        device = self.__openedDevice
        array = numpy.zeros(numberOfWords, dtype = numpy.int32)
        #numberOfWords = numberOfWords * 4
        if(type(offset) != int):
            print "offset is not type int"
            return None

        if(type(registerName) == str):
            device.readRaw(registerName, array, numberOfWords*4, offset)

        device.readRaw(offset, array, numberOfWords*4, bar)
        return array

    def writeRaw(self, offset, array, registerName=None, bar=0):
        device = self.__openedDevice
        wordsToWrite = array.size * 4
        if(wordsToWrite == 0 ):
            print "Nothing to write"
            return None

        if(type(offset) != int):
            print "offset is not type int"
            return None

        if(array.dtype != numpy.int32):
            print "expecting a numpy.int32 type array"
            return None

        if(type(registerName) == str):
            device.writeRaw(registerName, array, wordsToWrite, offset)
        device.writeRaw(offset, array, wordsToWrite, bar)



        
    def wrapper(self, regoffset=0, numpyArray=numpy.array([1], dtype=numpy.int32) ,
            size=1, bar=0):
        print self.__openedDevice
        #device = Device()
        #device.readRaw(regoffset, numpyArray, size, bar)

