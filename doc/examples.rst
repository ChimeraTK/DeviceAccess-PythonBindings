Examples
========

This page contains practical examples demonstrating common usage patterns with the DeviceAccess Python bindings.
The actual example code, and the `map` and `dmap` files used in this example can be found in the `tests` folder of the source distribution. The content is also listed in the :ref:`Used Map Files section <used_map_files_section>` section below.

.. _basic_example_python:

Basic Scalar Register Access
-----------------------------

Reading and writing a single register value:

.. literalinclude:: ../tests/testDocExamples.py
   :pyobject: TestDocExamples.simpleScalarAccess
   :lines: 2-
   :dedent: 4


.. _array_example_python:

Working with 1D Accessors
-------------------

Reading and processing array data:

.. literalinclude:: ../tests/testDocExamples.py
   :pyobject: TestDocExamples.simpleOneDAccess
   :lines: 2-
   :dedent: 4


.. _device_map_example_python:

Using Different Device Backends
--------------------------------

The ChimeraTK library supports multiple backends. Configure them in your device map file:

.. code-block:: text

   # Device map example
    AMC_PCIe    (xdma:xdma/slot6?map=device.map)
    SCPI_Dev    (CommandBasedTCP:lab_dev?map=hw_prep.json&port=50000)
    OPCUADev    (opcua:192.168.1.101?port=16664)
    DummyDev    (dummy?map=device.map)

Then use them the same way in your Python code:

.. code-block:: python

   import deviceaccess

   # Open different backend devices with same API
   dummy = deviceaccess.Device("DummyDev")
   pcie = deviceaccess.Device("AMC_PCIe")
   scpi = deviceaccess.Device("SCPI_Dev")
   opcua = deviceaccess.Device("OPCUADev")

   # All use the same accessor interface
   for device in [dummy, pcie, scpi, opcua]:
       value = device.getScalarRegisterAccessor("MEASUREMENT")
       value.read()
       print(f"Value: {float(value)}")


.. _transfer_groups_python:

Synchronized Access with Transfer Groups
-----------------------------------------

Use transfer groups to read/write multiple registers atomically:

.. code-block:: python

   import deviceaccess

   device = deviceaccess.Device("MY_DEVICE")

   # Create a transfer group
   group = device.getTransferGroup()

   # Add registers to the group
   voltage = device.getScalarRegisterAccessor("VOLTAGE")
   current = device.getScalarRegisterAccessor("CURRENT")
   power = device.getScalarRegisterAccessor("POWER")

   group.addAccessor(voltage)
   group.addAccessor(current)
   group.addAccessor(power)

   # Read all at once
   group.read()

   # All values are from the same hardware snapshot
   print(f"V={float(voltage)}, I={float(current)}, P={float(power)}")

   # Write all at once
   voltage.write(230.0)
   current.write(10.0)
   power.write(2300.0)

   group.write()


.. _data_consistency_python:

Data Consistency Groups
-----------------------

For reading coherent data across multiple samples:

.. code-block:: python

   import deviceaccess

   device = deviceaccess.Device("MY_DEVICE")

   # Create a data consistency group
   consistency_group = device.getDataConsistencyGroup()

   # Add accessors for channels
   channels = []
   for i in range(4):
       channel = device.getScalarRegisterAccessor(f"CHANNEL_{i}")
       channels.append(channel)
       consistency_group.addAccessor(channel)

   # Read all channels with guaranteed consistency
   consistency_group.read()

   # Process the consistent data
   values = [float(ch) for ch in channels]
   print(f"Channel values: {values}")


Error Handling
--------------

Robust code includes proper error handling:

.. code-block:: python

   import deviceaccess

   try:
       device = deviceaccess.Device("MY_DEVICE")
   except deviceaccess.DoocsException as e:
       print(f"Failed to open device: {e}")
       exit(1)

   try:
       register = device.getScalarRegisterAccessor("MEASUREMENT")
       register.read()
       value = float(register)
       print(f"Read value: {value}")
   except deviceaccess.DoocsException as e:
       print(f"Read operation failed: {e}")
   except ValueError as e:
       print(f"Type conversion failed: {e}")


Monitoring Values Over Time
----------------------------

Read values periodically:

.. code-block:: python

   import deviceaccess
   import time

   device = deviceaccess.Device("MY_DEVICE")
   temperature = device.getScalarRegisterAccessor("TEMPERATURE")

   readings = []
   for i in range(10):
       temperature.read()
       value = float(temperature)
       readings.append(value)
       print(f"Reading {i+1}: {value} °C")

       if i < 9:
           time.sleep(1.0)

   print(f"Average: {sum(readings) / len(readings)} °C")


Batch Operations
----------------

Efficiently perform multiple operations:

.. code-block:: python

   import deviceaccess

   device = deviceaccess.Device("MY_DEVICE")

   # Get multiple accessors at once
   registers = {
       "voltage": device.getScalarRegisterAccessor("VOLTAGE"),
       "current": device.getScalarRegisterAccessor("CURRENT"),
       "frequency": device.getScalarRegisterAccessor("FREQUENCY"),
   }

   # Read all
   for accessor in registers.values():
       accessor.read()

   # Process results
   status = {name: float(accessor) for name, accessor in registers.items()}
   print(f"Device status: {status}")


Integration with NumPy and Pandas
---------------------------------

Working with scientific Python libraries:

.. code-block:: python

   import deviceaccess
   import pandas as pd
   import numpy as np

   device = deviceaccess.Device("MY_DEVICE")

   # Collect time-series data
   data = []
   for i in range(100):
       waveform = device.getArrayRegisterAccessor("WAVEFORM")
       waveform.read()

       data.append({
           'timestamp': i,
           'mean': np.mean(waveform[:]),
           'std': np.std(waveform[:]),
           'min': np.min(waveform[:]),
           'max': np.max(waveform[:]),
       })

   # Create DataFrame for analysis
   df = pd.DataFrame(data)
   print(df.describe())

.. _used_map_files_section:

Used Map Files
--------------

.. literalinclude:: ../tests/documentationExamples/someCrate.dmap
   :caption: Example Crate dMap File

.. literalinclude:: ../tests/documentationExamples/someDummyModule.map
   :caption: Example Module Map File


See Also
--------

* :doc:`user_guide` for in-depth explanations
* :doc:`api_reference` for complete API documentation
* :doc:`faq` for common questions
