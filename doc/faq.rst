Frequently Asked Questions
==========================


Installation and Setup
----------------------

**Q: How do I install the Python bindings?**

A: See the :doc:`getting_started` guide for detailed installation instructions.
   Quick version: install the distribution packages as described there, or build from source with CMake.


**Q: What Python versions are supported?**

A: Python 3.12 and higher are supported. Check the project's CI/CD configuration
   for the currently tested versions.


**Q: I'm getting ImportError when trying to import deviceaccess**

A: This usually means the package isn't installed or not in your Python path.
   Try: ``python -c "import deviceaccess; print(deviceaccess.__file__)"``

   If that fails, reinstall the package. See :doc:`troubleshooting` for more help.


Basic Usage
-----------

**Q: How do I read a value from a register?**

A: Create an accessor and call ``read()``:

   .. code-block:: python

      accessor = device.getScalarRegisterAccessor(float, "REGISTER_NAME")
      accessor.read()
      value = float(accessor)


**Q: When do I need to call read() or write()?**

A: - Call ``read()`` to transfer data **from hardware to the local buffer**
   - Call ``write()`` to transfer data **from the local buffer to hardware**
   - Between calls, you're working with the local buffer (no hardware communication)

   .. code-block:: python

      # Read from hardware
      accessor.read()

      # Work with local copy (fast, no hardware access)
      my_value = float(accessor)

      # Modify local copy
      accessor.set(my_value * 2)  # Or use accessor.write()

      # Write back to hardware
      accessor.write()


**Q: What's the difference between read/write and the accessor value?**

A: - ``read()`` / ``write()`` - Transfer data with hardware
   - Accessing the accessor value - Work with local buffer only

   .. code-block:: python

      # These don't talk to hardware:
      accessor.set(42.0)  # Modifies local buffer
      value = float(accessor)  # Reads local buffer

      # This transfers data:
      accessor.read()  # Get latest from hardware
      accessor.write()  # Send to hardware


**Q: Can I modify an accessor and then write it?**

A: Yes, but methods vary by accessor type:

   .. code-block:: python

      # Scalar
      scalar = device.getScalarRegisterAccessor(float, "VALUE")
      scalar.set(42.0)
      scalar.write()

      # Array
      array = device.getOneDRegisterAccessor(float, "DATA")
      # Can be treated like a python list with numpy methods, incl. slices
      array[0] = 1.5
      array[1] = 2.5
      array.write()


Accessors and Type Conversion
------------------------------

**Q: How does the library handle type conversion?**

A: Automatic conversion happens when you request the register:

   .. code-block:: python

      aFloatValue = dev.getScalarRegisterAccessor(float, "SENSORS.TEMPERATURE")
      anIntList = device.getOneDRegisterAccessor(int, "SENSORS.WAVEFORM")


Device Maps
-----------

**Q: What is a device map file?**

A: A device map (`.dmap`) file describes the devices your application can access.
   It maps logical device names to hardware locations and backend specifications.

   See :ref:`Device Maps <Device Maps>` in the user guide for details.


**Q: How do I debug device map issues?**

A: - Check that the file exists and is readable
   - Verify the syntax is correct
   - Try opening the device with QtHardMon or Chai.


Transfer Groups
---------------

**Q: When should I use transfer groups?**

A: Use transfer groups when you need to:

   - Read multiple registers with guaranteed consistency
   - Write multiple registers atomically
   - Reduce hardware communication overhead



**Q: What's the difference between transfer groups and data consistency groups?**

A: - **Transfer Group**: Synchronizes multiple registers in a single read/write
   - **Data Consistency Group**: Provides consistency via VersionNumbers


**Q: Do transfer groups improve performance?**

A: Yes, if the backend supports it.


Performance and Optimization
-----------------------------

**Q: How can I improve performance for many reads?**

A: - Reuse accessors instead of creating new ones each time
   - Minimize how often you call read/write

   See :doc:`user_guide` for optimization strategies.


**Q: Is there an asynchronous API?**

A: The synchronous API is the standard. For asynchronous access you need to set up the device:

   .. code-block:: python

      dev.activateAsyncRead()


Compatibility and Versions
---------------------------

**Q: Is this compatible with my existing C++ code?**

A: Yes! The Python bindings wrap the C++ library, providing the same functionality
   through a Pythonic interface. The underlying hardware access is identical.


**Q: What backend devices are supported?**

A: Supported backends depend on your ChimeraTK installation. The Python bindings offer the same support as the C++ version of DeviceAccess.


**Q: Can I use old code written for the mtca4u module?**

A: Yes, the ``mtca4u`` module is still supported for compatibility.
   However, new code should use the ``deviceaccess`` module instead.


Data Handling
-------------

**Q: How do I work with array data?**

A: Use ``ArrayRegisterAccessor``:

   .. code-block:: python

      array = device.getOneDRegisterAccessor(int, "DATA")
      array.read()

      # Access elements
      first_val = array[0]
      last_val = array[-1]

      # Iterate
      for val in array[:]:
          print(val)

      # Convert to list or numpy array
      data = list(array[:])
      import numpy as np
      data = np.array(array[:])


**Q: Can I modify array data and write it back?**

A: Yes:

   .. code-block:: python

      array = device.getOneDRegisterAccessor(float, "DATA")
      array.read()

      # Modify
      for i in range(len(array)):
          array[i] = array[i] * 2

      # Write back
      array.write()


**Q: Can I use NumPy with the bindings?**

A: Yes! The bindings were written with NumPy as a use-case.

   .. code-block:: python

      import numpy as np

      array = device.getOneDRegisterAccessor(int, "DATA")
      array.read()

      # Convert to NumPy
      data = np.array(array[:])

      # Analyze and modify
      processed = np.fft.fft(data)

      # Write back
      for i, val in enumerate(processed):
          array[i] = val
      array.write()


Still Have Questions?
---------------------

- Check the :doc:`examples` for practical code samples
- Review the :doc:`user_guide` for in-depth explanations
- See :doc:`troubleshooting` for problem-solving
- Check the API reference for class and method documentation
- Look at the `ChimeraTK documentation <https://chimeratk.github.io/>`_
