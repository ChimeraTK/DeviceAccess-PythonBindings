import mtca4udeviceaccess 
import numpy
import sys
import os

__version__ = "${${PROJECT_NAME}_VERSION}"

#http://stackoverflow.com/questions/4219717/how-to-assert-output-with-nosetest-unittest-in-python
def get_info(outputStream=sys.stdout):
  """ prints details about the module and the deviceaccess library
  against which it was linked
  
  Parameters
  ----------
  outputStream: optional
    default: sys.stdout
  
  Returns
  -------
  
  Examples
  --------
  >>> import mtca4u
  >>> mtca4u.get_info()
  mtca4uPy v${${PROJECT_NAME}_VERSION}, linked with mtca4u-deviceaccess v${mtca4u-deviceaccess_VERSION}
    
  """
  outputStream.write("mtca4uPy v${${PROJECT_NAME}_VERSION}, linked with mtca4u-deviceaccess v${mtca4u-deviceaccess_VERSION}")

def set_dmap_location(dmapFileLocation):
  """ Sets the location of the dmap file to use
  
  The library will check the user specified device Alias names (when creating
  devices) in this dmap file. Once set, the library will look at this dmap file
  through out the program lifetime. This is true until a new dmap file is
  set again using set_dmap_location
  
  Parameters
  ----------
  dmapFileLocation: string
    Path to the desired dmap file.
  
  Returns
  -------
  
  Examples
  --------
    >>> import mtca4u
    >>> mtca4u.set_dmap_location("../my_example_dmap_file.dmap")
    >>> device = mtca4u.Device("my_card") # my_card is a alias in my_example_dmap_file.dmap
   
  See Also
  --------
  get_dmap_location: View the current dmap file which the library uses for device name (alias) lookup.
  Device : Open device using specified alias names or using device id and mapfile 

  """
  #os.environ["DMAP_FILE"] = dmapFileLocation
  mtca4udeviceaccess.setDmapFile(dmapFileLocation)
  
  
  
def get_dmap_location():
  """ Get the dmap file which is currently in use by the library.
  
  Method returns the file path of the dmap file the library currently uses.
  This is the dmap file the library uses to look up the device name(alias) and
  its details
    
  Parameters
  ----------
  
  Returns
  -------
  string: File path of the dmap file the library currently uses.
  
  Examples
  --------
    >>> import mtca4u
    >>> mtca4u.set_dmap_location("../my_example_dmap_file.dmap")
    >>> dmapPath = mtca4u.get_dmap_location()
    >>> print dmapPath # prints '../my_example_dmap_file.dmap'
    
  See Also
  --------
  set_dmap_location : set dmap file path for the library.
  Device : Open device using specified alias names or using device id and mapfile
   
  """
  return mtca4udeviceaccess.getDmapFile()

  
class Device():
  """ Construct Device from user provided device information
  
  This constructor is used to open a device listed in the dmap file.
  
  Parameters
  ----------
  alias : str
    The device alias/name in the dmap file for the hardware

  Examples
  --------
  Creating a device using dmap file:
    >>> import mtca4u
    >>> mtca4u.set_dmap_location("../my_example_dmap_file.dmap")
    >>> device = mtca4u.Device("my_card") # my_card is a alias in my_example_dmap_file.dmap
  """
  def __init__(self, *args):
    # define a dictiWe do not wantonary to hold multiplexed data accessors
    # TODO: Limit the number of entries this dictionary can hold,
    # We do not want to hold on to too much ram
    self.__accsessor_dictionary = {}
    
    if len(args) == 2:
      deviceFile = args[0]
      mapFile = args[1]
      self.__openedDevice = mtca4udeviceaccess.createDevice(deviceFile, mapFile)
      self.__printDeprecationWarning(deviceFile, mapFile) 
      
    elif len(args) == 1:
      cardAlias = args[0]
      if(get_dmap_location() == ''):
          self.__throwDmapFilePathNotSetException(cardAlias)
      self.__openedDevice = mtca4udeviceaccess.createDevice(cardAlias)
            
    else:
      raise SyntaxError("Syntax Error: please see help(mtca4u.Device) for usage instructions.")

  def read(self, moduleName, registerName, numberOfElementsToRead=0,
            elementIndexInRegister=0):
    """ Reads out Fixed point converted values from the opened device
    
    This method uses the map file to return Fixed Point converted values from a
    register. It can read the whole register or an arbitary number of register
    elements. Data can also be read from an offset within the register (through
    the 'elementIndexInRegister' parameter).
    
    Parameters
    ----------
    moduleName : str
      The name of the device module to which the register belongs to. If the
      register is not contained in a  module, then provide an empty string as
      the parameter value.
       
    registerName : str
      The name of the register to read from.
      
    numberOfElementsToRead : int, optional 
      Specifies the number of register elements that should
      be read out. The width and fixed point representation of the register
      element are internally obtained from the map file.
      
      The method returns all elements in the register if this parameter is
      ommitted or when its value is set as 0.
       
      If the value provided as this parameter exceeds the register size, an
      array with all elements upto the last element is returned
      
    elementIndexInRegister : int, optional
      This is a zero indexed offset from the first element of the register. When
      an elementIndexInRegister parameter is specified, the method reads out
      elements starting from this element index. The elemnt at the index
      position is included in the read as well.

    Returns
    -------
    readoutValues: numpy.array, dtype == numpy.float64
      The return type for the method is a 1-Dimensional numpy array with
      datatype numpy.float64. The returned numpy.array would either contain all
      elements in the register or only the number specified by the
      numberOfElementsToRead parameter
     
    Examples
    --------
    register "WORD_STATUS" is 1 element long..
      >>> import mtca4u
      >>> mtca4u.set_dmap_location("../my_example_dmap_file.dmap")
      >>> boardWithModules = mtca4u.Device("device_name")
      >>> boardWithModules.read("BOARD", "WORD_STATUS")
      array([15.0], dtype=float64)
      
    
    register "WORD_CLK_MUX" is 4 elements long.
      >>> import mtca4u
      >>> device = mtca4u.Device("device_name")
      >>> device.read("", "WORD_CLK_MUX")
      array([15.0, 14.0, 13.0, 12.0], dtype=float64)
      >>> device.read("", "WORD_CLK_MUX", 0)
      array([15.0, 14.0, 13.0, 12.0], dtype=float64)
    read out select number of elements from specified locations:  
      >>> device.read("", "WORD_CLK_MUX", 1)
      array([15.0], dtype=float64)
      >>> device.read("", "WORD_CLK_MUX", 1, 2 )
      array([13.0], dtype=float64)
      >>> device.read("", "WORD_CLK_MUX", 0, 2 )
      array([13.0, 12.0], dtype=float64)
      >>> device.read("", "WORD_CLK_MUX", 5, 2 )
      array([13.0, 12.0], dtype=float64)
      >>> device.read("", "WORD_CLK_MUX", numberOfElementsToRead=1, elementIndexInRegister=2 )
      array([13.0], dtype=float64)
      >>> device.read("", "WORD_CLK_MUX", elementIndexInRegister=2 )
      array([13.0, 12.0], dtype=float64)
    
    
    See Also
    --------
    Device.read_raw : Read in 'raw' bit values from a device register 

    """
    
    registerPath = moduleName + '/' + registerName 
    registerAccessor = self.__openedDevice.get1DAccessor_double(registerPath, 
                                                                numberOfElementsToRead,
                                                                elementIndexInRegister)
    
    registerSize = registerAccessor.getNumElements();
    array = numpy.empty(registerSize, numpy.double)
    registerAccessor.read(array)
    
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
      If module name is not applicable to the register, then provide an empty
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
    
    Examples
    --------
    register "WORD_STATUS" is 1 element long and belongs to module "BOARD".
      >>> import mtca4u
      >>> mtca4u.set_dmap_location("../my_example_dmap_file.dmap")
      >>> boardWithModules = mtca4u.Device("device_name")
      >>> boardWithModules.write("BOARD", "WORD_STATUS", 15)
      >>> boardWithModules.write("BOARD", "WORD_STATUS", 15.0)
      >>> boardWithModules.write("BOARD", "WORD_STATUS", [15])
      >>> boardWithModules.write("BOARD", "WORD_STATUS", [15.0])
      >>> dataToWrite = numpy.array([15.0])
      >>> boardWithModules.write("BOARD", "WORD_STATUS", dataToWrite)
    
    register "WORD_CLK_MUX" is 4 elements long.
      >>> import mtca4u
      >>> mtca4u.set_dmap_location("../my_example_dmap_file.dmap")
      >>> device = mtca4u.Device("device_name")
      >>> dataToWrite = numpy.array([15.0, 14.0, 13.0, 12.0])
      >>> device.write("", "WORD_CLK_MUX", dataToWrite)
      >>> dataToWrite = numpy.array([13, 12])
      >>> device.write("", "WORD_CLK_MUX", dataToWrite, 2)
      >>> device.write("", "WORD_CLK_MUX", 2.78) # writes value to first element of register
      >>> device.write("", "WORD_CLK_MUX", 10, elementIndexInRegister=3)
      
    See Also
    --------
    Device.write_raw : Write 'raw' bit values to a device register
    
    """
    # get register accessor
    data = numpy.array(dataToWrite)
    numberOfElementsToWrite = data.size
    if numberOfElementsToWrite == 0:
      return
    
    registerPath = moduleName + '/' + registerName
    arguments = (registerPath, 
                 numberOfElementsToWrite,
                 elementIndexInRegister)


    device = self.__openedDevice
    dtype = data.dtype
    if(dtype == numpy.int32):
        accessor = device.get1DAccessor_int32(*arguments)
    elif(dtype == numpy.int64):
        accessor = device.get1DAccessor_int64(*arguments)
    elif(dtype == numpy.float32):
        accessor = device.get1DAccessor_float(*arguments)
    elif(dtype == numpy.float64):
        accessor = device.get1DAccessor_double(*arguments)
    else:
        raise RuntimeError("Data format used is unsupported")

    accessor.write(data)
    
  
  def read_raw(self, moduleName, registerName, numberOfElementsToRead=0, 
              elementIndexInRegister=0):
    """ Returns 'raw values' (Without fixed point conversion applied) from a device's register
    
    This method returns the raw bit values contained in the queried register.
    The returned values are not Fixed Point converted, but direct binary values
    contained in the register elements.
    
    Parameters
    ----------
    moduleName : str
      The name of the device module to which the register to read from belongs.
      If module name is not applicable to the register, then provide an empty
      string as the parameter value.
      
    registerName : str
      The name of the device register to read from.
      
    numberOfElementsToRead : int, optional
      Specifies the number of register elements that should be read out.
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
      >>> import mtca4u
      >>> mtca4u.set_dmap_location("../my_example_dmap_file.dmap")
      >>> boardWithModules = mtca4u.Device("device_name")
      >>> boardWithModules.read_raw("BOARD", "WORD_STATUS")
      array([15], dtype=int32)
    
    register "WORD_CLK_MUX" is 4 elements long.
      >>> import mtca4u
      >>> mtca4u.set_dmap_location("../my_example_dmap_file.dmap")
      >>> device = mtca4u.Device("device_name")
      >>> device.read_raw("", "WORD_CLK_MUX")
      array([15, 14, 13, 12], dtype=int32)
      >>> device.read_raw("", "WORD_CLK_MUX", 0)
      array([15, 14, 13, 12], dtype=int32)
      >>> device.read_raw("", "WORD_CLK_MUX", 1)
      array([15], dtype=int32)
      >>> device.read_raw("", "WORD_CLK_MUX", 1, 2 )
      array([13], dtype = int32)
      >>> device.read_raw("", "WORD_CLK_MUX", 0, 2 )
      array([13, 12], dtype=int32)
      >>> device.read_raw("", "WORD_CLK_MUX", numberOfElementsToRead=1, elementIndexInRegister=2 )
      array([13], dtype=int32)
      >>> device.read_raw("", "WORD_CLK_MUX", elementIndexInRegister=2 )
      array([13, 12], dtype=int32)
    
    See Also
    --------
    Device.read : Read in Fixed Point converted bit values from a device register

    """
    registerPath = moduleName + '/' + registerName 
    registerAccessor = self.__openedDevice.getRaw1DAccessor(registerPath, 
                                                            numberOfElementsToRead,
                                                            elementIndexInRegister)
    
    registerSize = registerAccessor.getNumElements();
    array = numpy.empty(registerSize, numpy.int32)
    registerAccessor.read(array)
    return array
  
  def write_raw(self, moduleName, registerName, dataToWrite,
      elementIndexInRegister=0):
    """ Write raw bit values (no fixed point conversion applied) into the register
    
    Provides a way to put in a desired bit value into individual register
    elements. 
    
    Parameters
    ----------      
    moduleName : str
      The name of the device module that has the register we intend to write to.
      If module name is not applicable to the register, then provide an empty
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
    
    Examples
    --------
    register "WORD_STATUS" is 1 element long and is part of the module "BOARD".
      >>> import mtca4u
      >>> mtca4u.set_dmap_location("../my_example_dmap_file.dmap")
      >>> boardWithModules = mtca4u.Device("device_name")
      >>> dataToWrite = numpy.array([15], dtype=int32)
      >>> boardWithModules.write_raw("BOARD", "WORD_STATUS", dataToWrite)
    
    register "WORD_CLK_MUX" is 4 elements long.
      >>> import mtca4u
      >>> mtca4u.set_dmap_location("../my_example_dmap_file.dmap")
      >>> device = mtca4u.Device("device_name")
      >>> dataToWrite = numpy.array([15, 14, 13, 12], dtype=int32)
      >>> device.write_raw("", "WORD_CLK_MUX", dataToWrite)
      >>> dataToWrite = numpy.array([13, 12], dtype=int32)
      >>> device.write_raw("MODULE1", "WORD_CLK_MUX", dataToWrite, 2)
        
    See Also
    --------
    Device.write : Write values that get fixed point converted to the device
    
    """

    self.__checkAndExitIfArrayNotInt32(dataToWrite)

    numberOfElementsToWrite = dataToWrite.size
    if numberOfElementsToWrite == 0:
        return
    
    registerPath = moduleName + '/' + registerName
    accessor = self.__openedDevice.getRaw1DAccessor(registerPath, 
                                                    numberOfElementsToWrite,
                                                    elementIndexInRegister)
    accessor.write(dataToWrite)
  
  
  def read_dma_raw(self, moduleName, DMARegisterName, numberOfElementsToRead=0, 
                 elementIndexInRegister=0):
    """ Read in Data from the DMA region of the card
    
    This method can be used to fetch data copied to a dma memory block. The
    method assumes that the device maps the DMA memory block to a register made
    up of 32 bit elements.
    
    
    .. note:: Deprecated since 1.0.0; use Device.read_raw instead.
          
          
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
    Use Device.read_raw:
    In the example, register "AREA_DMA_VIA_DMA" is the DMA mapped memory made up of 32 bit elements.
      >>> import mtca4u
      >>> mtca4u.set_dmap_location("../my_example_dmap_file.dmap")
      >>> device = mtca4u.Device("device_name")
      >>> device.read__raw("", "AREA_DMA_VIA_DMA", 10)
      array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81], dtype=int32)

    See Also
    --------
    Device.read_raw : Use this method for the same purpose instead.
    """
    
    return self.read_raw(moduleName, DMARegisterName, 
                  numberOfElementsToRead, 
                  elementIndexInRegister)
    
    
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
    "DMA" is the Multiplexed data region name. This region is defined by 'AREA_MULTIPLEXED_SEQUENCE_DMA' in the mapfile.
      >>> import mtca4u
      >>> mtca4u.set_dmap_location("../my_example_dmap_file.dmap")
      >>> device = mtca4u.Device("device_name")
      >>> device.read_sequences("", "DMA")
      array([[   0.,    1.,    4.,    9.,   16.],
             [  25.,   36.,   49.,   64.,   81.],
             [ 100.,  121.,  144.,  169.,  196.],
             [ 225.,  256.,  289.,  324.,  361.]
             [ 400.,  441.,  484.,  529.,  576.]], dtype=float32)
             
    Each column of the 2D matrix represents an extracted sequence:
     >>> data = device.read_sequences("", "DMA")
     >>> adc0_values = data[:,0] # array([0., 25., 100., 225., 400.])
     >>> adc1_values = data[:,1] # array([1., 36., 49., 64., 81.])
     >>> adc3_values = data[:,3] # array([9., 64., 169., 324., 529.])
             
    """
    
    registerPath = moduleName + '/' + regionName
    accessor = self.__openedDevice.get2DAccessor(registerPath)
    
    # readFromDevice fetches data from the card to its intenal buffer of the
    # c++ accessor
    numberOfSequences = accessor.getNChannels()
    numberOfBlocks = accessor.getNElementsPerChannel()
    array2D = self.__create2DArray(numpy.float32, numberOfBlocks,
                                   numberOfSequences)
    
    accessor.read(array2D)
    return array2D


# Helper methods below    
    
  def __exitIfSuppliedIndexIncorrect(self, registerAccessor, elementIndexInRegister):
    registerSize = registerAccessor.getNumElements()
    if(elementIndexInRegister >= registerSize):
      if(registerSize == 1):
        # did this for displaying specific error string without the range when
        # there is only one element in the register
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

  def __printDeprecationWarning(self, deviceFile, mapFile):
      deviceFileName = self.__extractNameFromDeviceFile(deviceFile)
      
      print("*************************************************************************************************")
      print(" >>> mtca4u.Device('" + deviceFile + "', '" + mapFile + "')")
      print(" The above usage for device creation will be phased out")
      print("                                                                            ")
      print(" Please consider using a dmap file for device creation.                     ")
      print(" Instructions:                                                              ")
      print(" - Create dmap file: <your_dmapfile_name_goes_here>.dmap                    ")
      print("                                                                            ")
      print(" - Add this line to your dmap file:                                         ")
      print("     <your_card_alias_goes_here> sdm://./pci:" + deviceFileName + "; " + mapFile)
      print("                                                                            ")
      print(" - Tell the library about the dmap file                                     ")
      print("     >>> mtca4u.set_dmap_location('your_dmapfile_name.dmap')                ")
      print("                                                                            ")
      print(" - Create your device                                                       ")
      print("     >>> device = mtca4u.Device('your_card_alias')                          ")
      print("*************************************************************************************************")
    

    
  def __extractNameFromDeviceFile(self, deviceFile):
      index = (deviceFile.rfind('/')) + 1
      return deviceFile[index:]
  
  def __throwDmapFilePathNotSetException(self, cardAlias):
      exceptionMessage = "Could not find a dmapfile. Please specify a dmap file to use.\n Can be done using mtca4u.set_dmap_location. See help(mtca4u.set_dmap_location)." 
      
      raise RuntimeError(exceptionMessage)
  