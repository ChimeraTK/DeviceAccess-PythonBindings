User Guide
==========

This guide covers important concepts and best practices for using the ChimeraTK DeviceAccess Python bindings.


Understanding Accessors
------------------------

Accessors are the primary interface for reading and writing device registers. They encapsulate:

* A reference to a specific register
* A local buffer holding the current value
* Type information and conversion logic
* Synchronization with hardware


Accessor Lifecycle
~~~~~~~~~~~~~~~~~~~

1. **Obtain**: Get an accessor from a device for a specific register
2. **Transfer**: Use ``read()`` to get data from hardware or ``write()`` to send data
3. **Access**: Read/write the local buffer without hardware communication
4. **Repeat**: Transfer more data as needed

.. code-block:: python

   import deviceaccess

   device = deviceaccess.Device("MY_DEVICE")

   # Obtain: Get accessor
   value = device.getScalarRegisterAccessor("VALUE")

   # Transfer: Read from hardware
   value.read()

   # Access: Work with local buffer (no hardware communication)
   current = float(value)
   doubled = current * 2

   # Modify and transfer back
   value.write(doubled)


Accessor Types
~~~~~~~~~~~~~~

**ScalarRegisterAccessor**: Single values

.. code-block:: python

   scalar = device.getScalarRegisterAccessor("VOLTAGE")
   scalar.read()
   print(float(scalar))  # Behaves like a float
   scalar.write(42.0)


**ArrayRegisterAccessor**: Array of values

.. code-block:: python

   array = device.getArrayRegisterAccessor("SPECTRUM")
   array.read()
   print(len(array))      # Length of array
   print(array[0])        # First element
   for val in array[:]:   # Iterate over all values
       print(val)


**NDRegisterAccessor**: Multi-dimensional arrays

.. code-block:: python

   # For 2D or higher dimensional data
   matrix = device.getNDRegisterAccessor("IMAGE")
   matrix.read()
   # Access like a nested array


Type Conversion
---------------

Automatic Type Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~

The bindings automatically convert between hardware and Python types:

.. code-block:: python

   # Hardware might store as raw integers, Python sees floats
   sensor = device.getScalarRegisterAccessor("TEMPERATURE")
   sensor.read()

   # These all work:
   temp_float = float(sensor)
   temp_int = int(sensor)
   temp_str = str(sensor)


Explicit Type Specification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When getting an accessor, specify the Python type you want:

.. code-block:: python

   # Read as double
   double_val = device.getScalarRegisterAccessor("VALUE")

   # Read as integer
   int_val = device.getScalarRegisterAccessor("COUNTER")


Handling Arrays
~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   array = device.getArrayRegisterAccessor("DATA")
   array.read()

   # Convert to NumPy for analysis
   data = np.array(array[:])

   # Modify and write back
   modified = data * 2
   for i, val in enumerate(modified):
       array[i] = val
   array.write()


Transfer Groups
---------------

Transfer groups enable atomic operations on multiple registers:

Motivation
~~~~~~~~~~

Without transfer groups, reading multiple registers could result in inconsistent data
if a register changes between reads.

.. code-block:: python

   # Problem: Values might change between reads
   voltage = device.getScalarRegisterAccessor("VOLTAGE")
   voltage.read()

   current = device.getScalarRegisterAccessor("CURRENT")
   current.read()
   # Current was just updated, but voltage is old!


Solution: Transfer Groups
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Solution: Read both at once
   group = device.getTransferGroup()

   voltage = device.getScalarRegisterAccessor("VOLTAGE")
   current = device.getScalarRegisterAccessor("CURRENT")

   group.addAccessor(voltage)
   group.addAccessor(current)

   group.read()  # Both updated atomically


Data Consistency Groups
-----------------------

For advanced scenarios with multiple samples or high-frequency updates:

.. code-block:: python

   consistency_group = device.getDataConsistencyGroup()

   # Add accessors you want consistent snapshots of
   register1 = device.getScalarRegisterAccessor("REGISTER_1")
   register2 = device.getScalarRegisterAccessor("REGISTER_2")

   consistency_group.addAccessor(register1)
   consistency_group.addAccessor(register2)

   # Read guarantees consistency across all accessors
   consistency_group.read()


Error Handling
--------------

Understanding Exceptions
~~~~~~~~~~~~~~~~~~~~~~~~~

The library raises exceptions for error conditions:

.. code-block:: python

   import deviceaccess

   try:
       device = deviceaccess.Device("MY_DEVICE")
   except deviceaccess.Exception as e:
       print(f"Error: {e}")

   try:
       register = device.getScalarRegisterAccessor("MISSING")
       register.read()
   except deviceaccess.DoocsException as e:
       print(f"Device error: {e}")


Common Exceptions
~~~~~~~~~~~~~~~~~

* ``DeviceException``: Device not found or unavailable
* ``DoocsException``: DOOCS backend specific errors
* ``NotImplemented``: Feature not supported by backend
* ``TimeoutException``: Operation took too long


Best Practices for Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import deviceaccess
   import time

   def safe_read(accessor, max_retries=3):
       """Read with retry logic"""
       for attempt in range(max_retries):
           try:
               accessor.read()
               return float(accessor)
           except deviceaccess.TimeoutException:
               if attempt < max_retries - 1:
                   time.sleep(0.1)  # Brief delay before retry
               else:
                   raise


Device Maps
-----------

Device maps define your hardware configuration.

Basic Syntax
~~~~~~~~~~~~

.. code-block:: text

   # Device map format
   # LABEL              BACKEND_SPECIFICATION

   # Dummy device for testing
   TEST_DEVICE        (dummy_name_prefix:?)

   # Real DOOCS device
   ACCELERATOR        (doocs://192.168.1.100)

   # Modbus TCP device
   SENSOR_ARRAY       (modbus://192.168.1.50?address_list=sensors.xml)


Best Practices
~~~~~~~~~~~~~~

* Use meaningful device labels
* Organize by system or subsystem
* Document non-obvious backend specifications
* Keep separate maps for testing and production
* Version control your device maps


Performance Considerations
--------------------------

Minimize Hardware Communication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Inefficient: Multiple reads
   for i in range(1000):
       register.read()
       print(float(register))

   # Efficient: Read once, work with value
   register.read()
   value = float(register)
   for i in range(1000):
       print(value)


Use Transfer Groups
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Multiple individual reads
   for r in registers:
       r.read()  # Each is a separate hardware operation

   # Transfer group: Single operation
   group = device.getTransferGroup()
   for r in registers:
       group.addAccessor(r)
   group.read()  # One hardware operation for all


Batch Operations
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Instead of:
   for value in values:
       register.write(value)

   # Consider: Buffer updates and write together
   register.write(final_value)


Thread Safety
-------------

The library provides synchronous, thread-safe operations, but there are considerations:

Thread Safety Guarantees
~~~~~~~~~~~~~~~~~~~~~~~~

* Device objects are thread-safe for read/write operations
* Accessors from the same device can be used concurrently
* Each read/write operation is atomic

.. code-block:: python

   import threading
   import deviceaccess

   device = deviceaccess.Device("MY_DEVICE")

   def read_thread():
       register = device.getScalarRegisterAccessor("VALUE")
       while True:
           register.read()
           print(f"Read: {float(register)}")

   def write_thread():
       register = device.getScalarRegisterAccessor("VALUE")
       register.write(42.0)

   # Both threads can safely access the device
   t1 = threading.Thread(target=read_thread)
   t2 = threading.Thread(target=write_thread)
   t1.start()
   t2.start()


Debugging
---------

Enabling Debug Output
~~~~~~~~~~~~~~~~~~~~~

Set environment variables to enable debug logging:

.. code-block:: bash

   export DEVICEACCESS_DEBUG=1
   python your_script.py


Checking Device Availability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import deviceaccess

   try:
       device = deviceaccess.Device("MY_DEVICE")
       print("Device opened successfully")
   except Exception as e:
       print(f"Cannot open device: {e}")


Testing Connections
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import deviceaccess

   device = deviceaccess.Device("MY_DEVICE")

   # Try a simple read
   try:
       test_register = device.getScalarRegisterAccessor("TEST_REGISTER")
       test_register.read()
       print(f"Connection OK, value: {float(test_register)}")
   except Exception as e:
       print(f"Connection failed: {e}")


Memory Management
-----------------

Accessors hold references to registers, so be mindful of:

* Creating many accessors for large arrays
* Long-lived accessor objects
* Memory usage in monitoring loops

.. code-block:: python

   # Good: Reuse accessors
   register = device.getScalarRegisterAccessor("VALUE")
   for i in range(1000):
       register.read()
       # Process...

   # Avoid: Creating accessors in loops
   for i in range(1000):
       register = device.getScalarRegisterAccessor("VALUE")
       register.read()


See Also
--------

* :doc:`examples` for practical patterns
* :doc:`api_reference` for complete API
* :doc:`faq` for common questions
* :doc:`troubleshooting` for problem solving
