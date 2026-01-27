Getting Started
===============

Installation
------------

Prerequisites
~~~~~~~~~~~~~

* Python 3.6 or higher
* CMake 3.16 or higher (for building from source)
* ChimeraTK DeviceAccess library installed

Using Package Manager
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # On Debian/Ubuntu systems
   sudo apt-get install python3-chimeratk-deviceaccess

From Source
~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/ChimeraTK/ChimeraTK-DeviceAccess-PythonBindings.git
   cd ChimeraTK-DeviceAccess-PythonBindings

   # Build and install
   mkdir build && cd build
   cmake ..
   make
   sudo make install


Your First Program
------------------

Let's create a simple program to read a value from a device.

Prerequisites
~~~~~~~~~~~~~

You'll need a device map file (``devices.dmap``) that describes your device:

.. code-block:: text

   (DEVICE_LABEL)   (URI)
   MY_DEVICE        (dummy_name_prefix:?)


Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   import deviceaccess

   # Open the device using its name from the device map
   device = deviceaccess.Device("MY_DEVICE")

   # Get an accessor for a scalar register
   temperature = device.getScalarRegisterAccessor("TEMPERATURE")

   # Read the value from hardware
   temperature.read()

   # Access the value (accessor acts like the data type)
   print(f"Temperature: {float(temperature)}")


Step-by-Step Explanation
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Import**: The ``deviceaccess`` module contains all necessary classes
2. **Device Creation**: ``Device()`` opens a connection to the hardware
3. **Get Accessor**: Accessors are type-safe handles to registers
4. **Read/Write**: ``read()`` and ``write()`` transfer data to/from hardware
5. **Data Access**: Accessors behave like the data they represent

.. note::

   All read and write operations are **synchronous** - they block until the operation completes.
   Check the :doc:`user_guide` for asynchronous patterns and advanced usage.


Working with Device Maps
------------------------

The device map file is crucial for telling the library about your devices.

Basic Format
~~~~~~~~~~~~

.. code-block:: text

   # Comments start with #
   # Format: DEVICE_NAME    BACKEND_SPECIFICATION

   MY_DEVICE           (dummy_name_prefix:?)
   REAL_DEVICE         (modbus://192.168.1.100?address_list=device.xml)


Finding Device Map Files
~~~~~~~~~~~~~~~~~~~~~~~~~

Device map files are typically:

* Located in your project's configuration directory
* Named with a ``.dmap`` extension
* Pointed to via environment variables or hardcoded paths
* Documented in your project's setup guide


Accessor Types
--------------

The library provides different accessor types for different data patterns:

ScalarRegisterAccessor
~~~~~~~~~~~~~~~~~~~~~~

For single values:

.. code-block:: python

   # Floating-point value
   voltage = device.getScalarRegisterAccessor("VOLTAGE")
   voltage.read()
   print(float(voltage))

   # Integer value
   count = device.getScalarRegisterAccessor("COUNTER")
   count.read()
   print(int(count))


ArrayRegisterAccessor
~~~~~~~~~~~~~~~~~~~~~

For arrays of values:

.. code-block:: python

   # Get an array accessor
   spectrum = device.getArrayRegisterAccessor("SPECTRUM")
   spectrum.read()

   # Access as list
   data = spectrum[:]
   print(f"Read {len(data)} values")


Next Steps
----------

Now that you have the basics:

* See :doc:`examples` for more real-world patterns
* Read the :doc:`user_guide` for deeper concepts
* Check the :doc:`api_reference` for complete API details
* Browse :doc:`faq` for common questions


Common Issues
~~~~~~~~~~~~~

* **Device not found**: Check that your device map file is accessible and correctly configured
* **Import errors**: Ensure the Python bindings are properly installed in your Python path
* **Permission denied**: You may need elevated privileges for certain hardware backends

See :doc:`troubleshooting` for more help.
