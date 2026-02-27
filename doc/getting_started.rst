Getting Started
===============

Installation
------------

Prerequisites
~~~~~~~~~~~~~

* Python 3.12 or higher, might work with older versions but not tested
* CMake 3.16 or higher (for building from source)
* ChimeraTK DeviceAccess library installed

Repository-Based Installation on Debian/Ubuntu systems
~~~~~~~~~~~~~~~~~~~~~

If you haven't already, add the public DOOCS Package Repository to your system, receive the DESY DOOCS key and add the DOOCS repository.

.. code-block:: bash

   wget -O - https://doocs-web.desy.de/pub/doocs/DOOCS-key.gpg.asc | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/doocs-keyring.gpg
   sudo sh -c 'echo "deb https://doocs-web.desy.de/pub/doocs  $(lsb_release -cs) main" > /etc/apt/sources.list.d/doocs.list'

Installation of the actual Python bindings package can then be done via apt, the package is named ``python3-mtca4upy``:

.. code-block:: bash

   sudo apt update && sudo apt-get install python3-mtca4upy

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
You will need a device with a map file, that can be referenced in a dmap with the respective backend. For testing purposes, you can use the dummy backend with a dummy device map entry like this:

.. literalinclude:: ../tests/documentationExamples/someCrate.dmap
   :language: text



.. note:: When testing application code, it is often beneficial not to rely on real hardware.
   ChimeraTK-DeviceAccess provides two backends for this purpose, the ChimeraTK::DummyBackend and the ChimeraTK::SharedDummyBackend.
   The DummyBackend emulates a devices' register space in application memory.
   The SharedDummyBackend allocates the registers in shared memory, so it can be access from multiple processes.
   E.g., QtHardMon or Chai can be used to stimulate and monitor a running application.
   Hence, these backends provide a generic way to test input-/output- operations on the application.

The following snippet gives a map file with a 32-bit scalar register and an 8-bit array register, that can be used with the dummy backends:

.. literalinclude:: ../tests/documentationExamples/someDummyModule.map
   :language: text


Basic Example
~~~~~~~~~~~~~

.. literalinclude:: ../tests/testDocExamples.py
   :pyobject: TestDocExamples.simpleScalarAccess
   :lines: 2-
   :dedent: 8


Step-by-Step Explanation
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Import**: The ``deviceaccess`` module contains all necessary classes and methods
2. **Initial Setup**: Set up dmap file name according to your configuration
3. **Device Creation**: ``Device()`` opens a connection to the hardware with the handle defined in the dmap file
4. **Get Accessor**: Accessors are type-safe handles to registers, regardless of the underlying hardware
5. **Read/Write**: ``read()`` and ``write()`` transfer data to/from hardware
6. **Data Access**: Accessors behave like the data they represent and can be used like Python types (e.g., float, int, list, even numpy arrays) after reading

.. note::

   By default, all read and write operations are **synchronous** - they block until the operation completes.
   Check the :doc:`user_guide` for asynchronous patterns and advanced usage.

Next Steps
----------

Now that you have the basics:

* See :doc:`examples` for more real-world patterns
* Read the :doc:`user_guide` for deeper concepts
* Check the :doc:`api_reference` for complete API details
* Browse :doc:`faq` for common questions
