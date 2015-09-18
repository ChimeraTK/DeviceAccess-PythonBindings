import mtcamappeddevice 
import numpy


class Device():
  """ This class represents the hardware device to access 
  
  This class can be used to open and acess the registers of a mapped device
  
  Parameters
  ----------
  deviceName : str
    The device file identifier for the hardware

  mapFile : str
    The location of the register mapped file for the hardware under
    consideration
  
  Examples
  --------
  >>> device = mtca4u.Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
  """
  
  def __init__(self, deviceName, mapFile):
    """ Constructor for the Device class
    """
    self.__openedDevice = mtcamappeddevice.createDevice(deviceName, mapFile)
    # define a dictionary to hold multiplexed data accessors
    # TODO: Limit the number of entries this dictionary can hold, because
    # multiplexed data accessors are currently memory expensive. We do not want
    # to hold on to too much memory
    self.__accsessor_dictionary = {}


  def read(self, moduleName, registerName, numberOfElementsToRead=0,
            elementIndexInRegister=0):
    """ Reads out Fixed point converted values from the opened mapped device
    
    This method uses the register mapping information to return Fixed Point
    converted versions of the values contained in a register. This method can be
    used to read in the whole register or an arbitary number of register
    elements from anywhere within the register (through the
    'elementIndexInRegister' parameter).
    
    Parameters
    ----------
    moduleName : str
      The name of the device module to which the register to read from belongs.
      If module name is not applicable to the device, then provide an empty
      string as the parameter value.
       
    registerName : str
      The name of the register to read from.
      
    numberOfElementsToRead : int, optional 
      Optional parameter specifying the number of register elements that should
      be read out. The width and fixed point representation of the register
      element are internally obtained from the mapping file.
      
      The method returns all elements in the register if this parameter is
      ommitted or when its value is set as 0. 
      
      If the value provided as this parameter exceeds the register size, an
      array will all elements upto the last element is returned
      
    elementIndexInRegister : int, optional
      This is a zero indexed offset from the first element of the register. When
      an elementIndexInRegister parameter is specified, the method reads out
      elements starting from this element index. The elemnt at the index
      position is included in the read as well.

    Returns
    -------
    readoutValues: numpy.array, dtype == numpy.float32
      The return type for the method is a 1-Dimensional numpy array with
      datatype numpy.float32. The returned numpy.array would either contain all
      elements in the register or only the number specified by the
      numberOfElementsToRead parameter
     
    Examples
    --------
    register "WORD_STATUS" is 1 element long..
      >>> boardWithModules = mtca4u.Device("/dev/llrfdummys4", "mapfiles/mtcadummy.map")
      >>> boardWithModules.read("BOARD", "WORD_STATUS")
      array([15.0], dtype=float32)
      
    
    register "WORD_CLK_MUX" is 4 elements long.
      >>> device = mtca4u.Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
      >>> device.read("", "WORD_CLK_MUX")
      array([15.0, 14.0, 13.0, 12.0], dtype=float32)
      >>> device.read("", "WORD_CLK_MUX", 0)
      array([15.0, 14.0, 13.0, 12.0], dtype=float32)
    reading beyond a valid register size returns all values:  
      >>> device.read("", "WORD_CLK_MUX", 5)
      array([15.0, 14.0, 13.0, 12.0], dtype=float32)
    read out select number of elements from specified locations:  
      >>> device.read("", "WORD_CLK_MUX", 1)
      array([15.0], dtype=float32)
      >>> device.read("", "WORD_CLK_MUX", 1, 2 )
      array([13.0], dtype=float32)
      >>> device.read("", "WORD_CLK_MUX", 0, 2 )
      array([13.0, 12.0], dtype=float32)
      >>> device.read("", "WORD_CLK_MUX", 5, 2 )
      array([13.0, 12.0], dtype=float32)
      >>> device.read("", "WORD_CLK_MUX", numberOfElementsToRead=1, elementIndexInRegister=2 )
      array([13.0], dtype=float32)
      >>> device.read("", "WORD_CLK_MUX", elementIndexInRegister=2 )
      array([13.0, 12.0], dtype=float32)
    
    
    See Also
    --------
    Device.read_raw : Read in 'raw' bit values from a device register 

    """
    
    registerAccessor = self.__openedDevice.getRegisterAccessor(moduleName,
                                                                registerName)
    # throw if element index  exceeds register size
    self.__exitIfSuppliedIndexIncorrect(registerAccessor, elementIndexInRegister)
    registerSize = registerAccessor.getNumElements();
    array = self.__createArray(numpy.float32, registerSize,
                                               numberOfElementsToRead,
                                               elementIndexInRegister)
    registerAccessor.read(array, array.size,
                          elementIndexInRegister)
    
    return array
  
  def write(self, moduleName, registerName, dataToWrite, elementIndexInRegister=0):
    """ Sets data into a desired register
    
    This method writes values into a register on the board. The method
    internally uses a fixed point converter that is aware of the register width
    on the device and its fractional representation. This Fixed point converter
    converts the input into corresponding Fixed Point representaions that fit
    into the decive register.
    
    Parameters
    ----------
    moduleName : str
      The name of the device module which has the register to write into.
      If module name is not applicable to the device, then provide an empty
      string as the parameter value.
      
    registerName : str
      Mapped name of the register to write to
      
    dataToWrite : int, float, \
    list of int/float, numpy.array(dtype numpy.float32/64), \
    numpy.array(dtype = numpy.int32/64) 
      The data to be written in to the register. it may be a numpy.float32/64 or a
      numpy.int32/64 array or a list with int or float values . Each value in this
      array represents an induvidual element of the register. dataToWrite may also
      take on int/float type when single vaues are passesed
       
    elementIndexInRegister : int, optional
      This is a zero indexed offset from the first element of the register. When
      an elementIndexInRegister parameter is specified, the method starts the
      write from this index
    
    Returns
    -------
      None: None
    
    Examples
    --------
    register "WORD_STATUS" is 1 element long and belongs to module "BOARD".
      >>> boardWithModules = mtca4u.Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
      >>> dataToWrite = numpy.array([15.0])
      >>> boardWithModules.write("BOARD", "WORD_STATUS")
    
    register "WORD_CLK_MUX" is 4 elements long.
      >>> device = mtca4u.Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
      >>> dataToWrite = numpy.array([15.0, 14.0, 13.0, 12.0])
      >>> device.write("", "WORD_CLK_MUX", dataToWrite)
      >>> dataToWrite = numpy.array([13, 12])
      >>> device.write("", "WORD_CLK_MUX", dataToWrite, 2)
      >>> device.write("WORD_CLK_MUX", 2.78)
      >>> device.write("WORD_CLK_MUX", 10, elementIndexInRegister=3)
      
    See Also
    --------
    Device.write_raw : Write 'raw' bit values to a device register
    
    """
    # get register accessor
    registerAccessor = self.__openedDevice.getRegisterAccessor(moduleName, 
                                                               registerName)
    self.__exitIfSuppliedIndexIncorrect(registerAccessor, elementIndexInRegister)
    arrayToHoldData = numpy.array(dataToWrite)
    numberOfElementsToWrite = arrayToHoldData.size
    registerAccessor.write(arrayToHoldData, numberOfElementsToWrite,
                            elementIndexInRegister)
  
  def read_raw(self, moduleName, registerName, numberOfElementsToRead=0, 
              elementIndexInRegister=0):
    """ Returns the raw values from a device's register
    
    This method returns the raw bit values contained in the queried register.
    The returned values are not Fixed Point converted, but direct binary values
    contained in the register elements.
    
    Parameters
    ----------
    moduleName : str
      The name of the device module to which the register to read from belongs.
      If module name is not applicable to the device, then provide an empty
      string as the parameter value.
      
    registerName : str
      The name of the device register to read from
      
    numberOfElementsToRead : int, optional
      Optional parameter specifying the number of register elements that should
      be read out.
      
      The method returns all elements in the register if this parameter is
      ommitted or when its value is set as 0.
      
      If the value provided as this parameter exceeds the register size, an
      array will all elements upto the last element is returned
    
    elementIndexInRegister : int, optional
      This is a zero indexed offset from the first element of the register. When
      an elementIndexInRegister parameter is specified, the method reads out
      elements starting from this element index. The element at the index
      position is included in the read as well.
    
    Returns
    -------
    readInRawValues: numpy.array, dtype == numpy.int32
      The method returns a numpy.int32 array containing the raw bit values of
      the register elements. The length of the array either equals the number of
      elements that make up the register or the number specified through the
      numberOfElementsToRead parameter
    
    Examples
    --------
    register "WORD_STATUS" is 1 element long.
      >>> boardWithModules = mtca4u.Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
      >>> boardWithModules.read_raw("BOARD", "WORD_STATUS")
      array([15], dtype=int32)
    
    register "WORD_CLK_MUX" is 4 elements long.
      >>> device = mtca4u.Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
      >>> device.read_raw("", "WORD_CLK_MUX")
      array([15, 14, 13, 12], dtype=int32)
      >>> device.read_raw("", "WORD_CLK_MUX", 0)
      array([15, 14, 13, 12], dtype=int32)
      >>> device.read_raw("", "WORD_CLK_MUX", 5)
      array([15, 14, 13, 12], dtype=int32)
      >>> device.read_raw("", "WORD_CLK_MUX", 1)
      array([15], dtype=int32)
      >>> device.read_raw("", "WORD_CLK_MUX", 1, 2 )
      array([13], dtype = int32)
      >>> device.read_raw("", "WORD_CLK_MUX", 0, 2 )
      array([13, 12], dtype=int32)
      >>> device.read_raw("", "WORD_CLK_MUX", 5, 2 )
      array([13, 12], dtype=int32)
      >>> device.read_raw("", "WORD_CLK_MUX", numberOfElementsToRead=1, elementIndexInRegister=2 )
      array([13], dtype=int32)
      >>> device.read_raw("", "WORD_CLK_MUX", elementIndexInRegister=2 )
      array([13, 12], dtype=int32)
    
    See Also
    --------
    Device.read : Read in Fixed Point converted bit values from a device register

    """
    # use wrapper aroung readreg
    registerAccessor = self.__openedDevice.getRegisterAccessor(moduleName, 
                                                               registerName)
    # throw if element index  exceeds register size
    self.__exitIfSuppliedIndexIncorrect(registerAccessor, elementIndexInRegister)
    registerSize = registerAccessor.getNumElements();
    array = self.__createArray(numpy.int32, registerSize, 
                               numberOfElementsToRead, elementIndexInRegister) 
    registerAccessor.readRaw(array, array.size, elementIndexInRegister)
    
    return array
  
  def write_raw(self, moduleName, registerName, dataToWrite,
      elementIndexInRegister=0):
    """ Write raw bit values into the register
    
    Provides a way to put in a desired bit value into individual register
    elements. 
    
    Parameters
    ----------      
    moduleName : str
      The name of the device module that has the register we intend to write to.
      If module name is not applicable to the device, then provide an empty
      string as the parameter value.
      
    registerName : str
      The name of the desired register to write into.
      
    dataToWrite : numpy.array, dtype == numpy.int32
     The array holding the bit values to be written into the register. The numpy
     array is expected to contain numpy.int32 values
     
    elementIndexInRegister : int, optional
      This is a zero indexed offset from the first element of the register. When
      an elementIndexInRegister parameter is specified, the method starts the
      write from this index
    
    Returns
    -------
    None: None
    
    Examples
    --------
    register "WORD_STATUS" is 1 element long and is part of the module "BOARD".
      >>> boardWithModules = mtca4u.Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
      >>> dataToWrite = numpy.array([15], dtype=int32)
      >>> boardWithModules.write_raw("BOARD", "WORD_STATUS")
    
    register "WORD_CLK_MUX" is 4 elements long.
      >>> device = mtca4u.Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
      >>> dataToWrite = numpy.array([15, 14, 13, 12], dtype=int32)
      >>> device.write_raw("", "WORD_CLK_MUX", dataToWrite)
      >>> dataToWrite = numpy.array([13, 12], dtype=int32)
      >>> device.write_raw("MODULE1", "WORD_CLK_MUX", dataToWrite, 2)
        
    See Also
    --------
    Device.write : Write values that get fixed point converted to the device
    
    """
    registerAccessor = self.__openedDevice.getRegisterAccessor(moduleName,  
                                                               registerName)
    self.__checkAndExitIfArrayNotInt32(dataToWrite)
    self.__exitIfSuppliedIndexIncorrect(registerAccessor, elementIndexInRegister)

    numberOfElementsToWrite = dataToWrite.size
    registerAccessor.writeRaw(dataToWrite, numberOfElementsToWrite,
                               elementIndexInRegister)
  
  
  def read_dma_raw(self, moduleName, DMARegisterName, numberOfElementsToRead=0, 
                 elementIndexInRegister=0):
    """ Read in Data from the DMA region of the card
    
    This method can be used to fetch data copied to a dma memory block. The
    method assumes that the device maps the DMA memory block to a register made
    up of 32 bit elements
    
    Parameters
    ----------
    moduleName : str
      The name of the device module that has the register we intend to write to.
      If module name is not applicable to the device, then provide an empty
      string as the parameter value.
      
    DMARegisterName : str
      The register name to which the DMA memory region is mapped
    
    numberOfElementsToRead : int, optional  
      This optional parameter specifies the number of 32 bit elements that have
      to be returned from the mapped dma register. When this parameter is not
      specified or is provided with a value of 0,  every  element in the DMA
      memory block is returned.
    
      If the value provided as this parameter exceeds the register size, an
      array with all elements upto the last element is returned
    
    elementIndexInRegister : int, optional
      This parameter specifies the index from which the read should commence.
      
    Returns
    -------
    arrayOfRawValues: numpy.array, dtype == numpy.int32
      The method returns a numpy.int32 array containing the raw bit values
      contained in the DMA register elements. The length of the array either
      equals the number of 32 bit elements that make up the whole DMA region or
      the number specified through the numberOfElementsToRead parameter
    
    Examples
    --------
    In the example, register "AREA_DMA_VIA_DMA" is the DMA mapped memory made up of 32 bit elements.
      >>> device = mtca4u.Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
      >>> device.read_dma_raw("", "AREA_DMA_VIA_DMA", 10)
      array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81], dtype=int32)
      >>> device.read_dma_raw("ModuleADC", "AREA_DMA_VIA_DMA", 10, 2 )
      array([4, 9, 16, 25, 36, 49, 64, 81, 100, 121], dtype=int32)
      >>> device.read_dma_raw("", "AREA_DMA_VIA_DMA", numberOfElementsToRead=10, elementIndexInRegister=2 )
      array([4, 9, 16, 25, 36, 49, 64, 81, 100, 121], dtype=int32)

    """
    
        # use wrapper aroung readreg
    registerAccessor = self.__openedDevice.getRegisterAccessor(moduleName, 
                                                               DMARegisterName)
    # throw if element index  exceeds register size
    self.__exitIfSuppliedIndexIncorrect(registerAccessor, elementIndexInRegister)
    registerSize = registerAccessor.getNumElements();
    array = self.__createArray(numpy.int32, registerSize, 
                               numberOfElementsToRead, elementIndexInRegister)
    registerAccessor.readDMARaw(array, array.size, 
                          elementIndexInRegister)
    
    return array
    
    
  def read_sequences(self, moduleName, regionName):
    """ Read in all sequences from a Multiplexed data Region
    
    This method returns the demultiplexed sequences in the memory area specified
    by regionName. The data is returned as a 2D numpy array with the coulums
    representing induvidual sequences
    
    Parameters
    ----------
    moduleName : str
      The name of the device module that has the register we intend to write to.
      If module name is not applicable to the device, then provide an empty
      string as the parameter value.
      
    regionName : str
      The name of the memory area containing the multiplexed data.
    
    Returns
    -------
    2DarrayOfValues: numpy.array, dtype == numpy.float32
      The method returns a 2D numpy.float32 array containing extracted
      induvidual sequences as the columns
    
    Examples
    --------
    In the example, register "DMA" is the Multiplexed data region name. This
    region is defined by 'AREA_MULTIPLEXED_SEQUENCE_DMA' in the mapfile.
      >>> device = mtca4u.Device("/dev/llrfdummys4","mapfiles/mtcadummy.map")
      >>> device.read_sequences("", "DMA")
      array([[   0.,    1.,    4.,    9.,   16.],
             [  25.,   36.,   49.,   64.,   81.],
             [ 100.,  121.,  144.,  169.,  196.],
             [ 225.,  256.,  289.,  324.,  361.]
             [ 400.,  441.,  484.,  529.,  576.]], dtype=float32)
             
    """
    
    # this key goes into the dictionary
    key = moduleName + "." + regionName
    muxedRegisterAccessor = self.__accsessor_dictionary.get(key)
    
    if (muxedRegisterAccessor == None):
      # This accsessor is not in our dictionary, created and add
      muxedRegisterAccessor = self.__openedDevice.getMultiplexedDataAccessor(moduleName, regionName)
      self.__accsessor_dictionary.update({key: muxedRegisterAccessor})

    # readFromDevice fetches data from the card to its intenal buffer of the
    # c++ accessor
    muxedRegisterAccessor.readFromDevice();
    numberOfSequences = muxedRegisterAccessor.getSequenceCount()
    numberOfBlocks = muxedRegisterAccessor.getBlockCount()
    array2D = self.__create2DArray(numpy.float32, numberOfBlocks,
                                   numberOfSequences)
    # Copy the data read in from the card into the prepared 2D numpy array
    muxedRegisterAccessor.populateArray(array2D)
    return array2D


# Helper methods below    
    
  def __exitIfSuppliedIndexIncorrect(self, registerAccessor, elementIndexInRegister):
    registerSize = registerAccessor.getNumElements()
    if(elementIndexInRegister >= registerSize):
      if(registerSize == 1):
        errorString = "Element index: {0} incorrect. Valid index is {1}"\
        .format(elementIndexInRegister, registerSize-1)
      else:
        errorString = "Element index: {0} incorrect. Valid index range is [0-{1}]"\
        .format(elementIndexInRegister, registerSize-1)

      raise ValueError(errorString)
      

  def __checkAndExitIfArrayNotFloat32(self, dataToWrite):
    self.__raiseExceptionIfNumpyArraydTypeIncorrect(dataToWrite, numpy.float32)
      
  
  def __checkAndExitIfArrayNotInt32(self, dataToWrite):
    self.__raiseExceptionIfNumpyArraydTypeIncorrect(dataToWrite, numpy.int32)
      
  
  def __raiseExceptionIfNumpyArraydTypeIncorrect(self, numpyArray, dType):
    if((type(numpyArray) != numpy.ndarray) or 
       (numpyArray.dtype != dType)):
      raise TypeError("Method expects values in a {0} " 
                      " numpy.array".format(dType))  
  
  def __getCorrectedElementCount(self, elementCountInRegister, numberOfelements,
                                  elementOffset):
    elementCountInRegister #=  registerAccessor.getNumElements()
    maxFetchableElements = elementCountInRegister - elementOffset
    correctedElementCount = numberOfelements if (numberOfelements != 0 and numberOfelements <= maxFetchableElements) else maxFetchableElements
    return correctedElementCount

  def __createArray(self, dType, numberOfElementsInRegister, numberOfElementsToRead, 
                    elementIndexInRegister):
    size = self.__getCorrectedElementCount(numberOfElementsInRegister, 
                                           numberOfElementsToRead, 
                                           elementIndexInRegister)
    array = numpy.empty(size, dtype = dType)
    return array

  def __create2DArray(self, dType, numberOfRows, numberOfColumns):
      array2D = numpy.empty((numberOfRows, numberOfColumns), dtype=dType)
      return array2D
