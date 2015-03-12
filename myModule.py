import mtcamappeddevice 
import numpy


class Device():

    def __init__(self, deviceName, mapFile):
        # exception handling is a kludge for now
        try:
            mtcamappeddevice.createDevice(5)
        except Exception as self.__storedArgErrException:
            pass

        try:
                self.__openedDevice = mtcamappeddevice.createDevice(deviceName)
        except Exception, e:            
            if(e.__class__ == self.__storedArgErrException.__class__):
                print "Device name and mapfile name are expected to be strings"

    def read(self, registerName, numberOf32bitWordsToRead=1,
            offsetFromRegisterBaseAddress=0):

        device = self.__openedDevice


        # find the size of array in case of mapped:
        #array = numpy.zeros(numberOfWords, dtype = numpy.int32)
        #numberOfWords = numberOfWords * 4
        #if(type(addressOffset) != int):
            #print "offset is not type int"
            #return None
#
        #if(type(registerName) == str):
            #device.readRaw(registerName, array, numberOfWords*4, addressOffset)
        #else:
            #device.readRaw(addressOffset, array, numberOfWords*4, bar)
        #return array

    def readRaw(self, registerName, numberOf32BitWordsToRead=1
            offsetFromRegisterBaseAddress=0):


    def write(self, registerName, dataToWrite, offsetFromRegisterBaseAddress=0):
        pass

    def writeRaw(self, registerName, dataToWrite,
            offsetFromRegisterBaseAddress=0):
        #device = self.__openedDevice
        #wordsToWrite = array.size * 4
        #if(wordsToWrite == 0 ):
            #print "Nothing to write"
            #return None

        #if(type(addressOffset) != int):
            #print "offset is not type int"
            #return None

        #if(array.dtype != numpy.int32):
            #print "expecting a numpy.int32 type array"
            #return None

        #if(type(registerName) == str):
            #device.writeRaw(registerName, array, wordsToWrite, addressOffset)
        #else:
            #device.writeRaw(addressOffset, array, wordsToWrite, bar)



        
    def wrapper(self, regoffset=0, numpyArray=numpy.array([1], dtype=numpy.int32) ,
            size=1, bar=0):
        print self.__openedDevice
        #device = Device()
        #device.readRaw(regoffset, numpyArray, size, bar)

