Frequently Asked Questions
==========================


Installation and Setup
----------------------

**Q: How do I install the Python bindings?**

A: See the :doc:`getting_started` guide for detailed installation instructions.
   Quick version: ``pip install chimeratk-deviceaccess`` or build from source with CMake.


**Q: What Python versions are supported?**

A: Python 3.6 and higher are supported. Check the project's CI/CD configuration
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

      accessor = device.getScalarRegisterAccessor("REGISTER_NAME")
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
      accessor.write(my_value * 2)  # Or use accessor.write()

      # Write back to hardware
      accessor.write()


**Q: What's the difference between read/write and the accessor value?**

A: - ``read()`` / ``write()`` - Transfer data with hardware
   - Accessing the accessor value - Work with local buffer only

   .. code-block:: python

      # These don't talk to hardware:
      accessor.write(42.0)  # Modifies local buffer
      value = float(accessor)  # Reads local buffer

      # This transfers data:
      accessor.read()  # Get latest from hardware
      accessor.write()  # Send to hardware


**Q: Can I modify an accessor and then write it?**

A: Yes, but methods vary by accessor type:

   .. code-block:: python

      # Scalar
      scalar = device.getScalarRegisterAccessor("VALUE")
      scalar.write(42.0)
      scalar.write()

      # Array
      array = device.getArrayRegisterAccessor("DATA")
      array[0] = 1.5
      array[1] = 2.5
      array.write()


Accessors and Type Conversion
------------------------------

**Q: How does the library handle type conversion?**

A: Automatic conversion happens when you access the value:

   .. code-block:: python

      register.read()

      float_val = float(register)      # Converts to float
      int_val = int(register)          # Converts to int
      str_val = str(register)          # Converts to string


**Q: Can I specify a type when getting an accessor?**

A: The accessor is obtained with the hardware type information already known.
   Type conversion happens at access time:

   .. code-block:: python

      # Get accessor for this hardware register
      register = device.getScalarRegisterAccessor("TEMPERATURE")
      register.read()

      # Convert to the type you want
      celsius = float(register)
      fahrenheit = celsius * 9 / 5 + 32


**Q: What happens if I try to convert to an incompatible type?**

A: You'll get a ``ValueError`` or similar exception. Always handle conversion errors:

   .. code-block:: python

      try:
          value = int(register)
      except (ValueError, TypeError) as e:
          print(f"Cannot convert: {e}")


Device Maps
-----------

**Q: What is a device map file?**

A: A device map (`.dmap`) file describes the devices your application can access.
   It maps logical device names to hardware locations and backend specifications.

   See :ref:`Device Maps <Device Maps>` in the user guide for details.


**Q: Where should I put my device map file?**

A: Typically in your project's configuration directory. You then tell the application
   where to find it, usually through an environment variable or configuration file.

   ``export DEVICE_MAP_FILE=/path/to/devices.dmap``


**Q: How do I debug device map issues?**

A: - Check that the file exists and is readable
   - Verify the syntax is correct (see the user guide)
   - Try opening a device and catch exceptions for error messages
   - Set debug environment variables (see troubleshooting)


Transfer Groups
---------------

**Q: When should I use transfer groups?**

A: Use transfer groups when you need to:

   - Read multiple registers with guaranteed consistency
   - Write multiple registers atomically
   - Reduce hardware communication overhead

   .. code-block:: python

      # All read together
      group = device.getTransferGroup()
      group.addAccessor(voltage)
      group.addAccessor(current)
      group.read()  # One operation


**Q: What's the difference between transfer groups and data consistency groups?**

A: - **Transfer Group**: Synchronizes multiple registers in a single read/write
   - **Data Consistency Group**: Provides consistency semantics across multiple accesses

   Use transfer groups for most cases. Data consistency groups are for advanced scenarios.


**Q: Do transfer groups improve performance?**

A: Yes, typically. Instead of multiple hardware operations (one per register),
   a transfer group uses a single operation for all registers.


Error Handling
--------------

**Q: What exceptions can the library raise?**

A: The main ones are:

   - ``DoocsException`` - DOOCS backend errors
   - ``DeviceException`` - Device access errors
   - ``TimeoutException`` - Operation timeout
   - ``NotImplemented`` - Feature not supported

   See the API reference for a complete list.


**Q: How should I handle read/write errors?**

A: Always wrap hardware operations in try-except:

   .. code-block:: python

      import deviceaccess

      try:
          register.read()
      except deviceaccess.TimeoutException:
          print("Read timed out")
      except deviceaccess.DoocsException as e:
          print(f"Device error: {e}")


**Q: The device opens but register access fails. What's wrong?**

A: Several possibilities:

   - Register name is wrong or doesn't exist
   - Device connection was lost
   - Hardware is offline or not responding
   - Permission issues with the device

   Enable debug logging and check the error message. See :doc:`troubleshooting`.


Performance and Optimization
-----------------------------

**Q: How can I improve performance for many reads?**

A: - Use transfer groups for multiple related registers
   - Reuse accessors instead of creating new ones each time
   - Minimize how often you call read/write
   - Consider caching values if they change infrequently

   See :doc:`user_guide` for optimization strategies.


**Q: Is there an asynchronous API?**

A: The synchronous API is the standard. For asynchronous access:

   - Use Python threading to run read/write in background threads
   - Consider thread pools for high-volume operations
   - Look into the ``threading`` or ``asyncio`` modules


**Q: How much memory do accessors use?**

A: Memory usage is proportional to the data size. Array accessors for large arrays
   will use more memory. Generally not a concern unless working with many large accessors.


Compatibility and Versions
---------------------------

**Q: Is this compatible with my existing C++ code?**

A: Yes! The Python bindings wrap the C++ library, providing the same functionality
   through a Pythonic interface. The underlying hardware access is identical.


**Q: What backend devices are supported?**

A: Supported backends depend on your ChimeraTK installation:

   - Dummy (for testing)
   - DOOCS (common in accelerators)
   - Modbus TCP/RTU
   - Others depending on your build

   Check your device map documentation or ChimeraTK docs for available backends.


**Q: Can I use old code written for the mtca4u module?**

A: Yes, the ``mtca4u`` module is still supported for compatibility.
   However, new code should use the ``deviceaccess`` module instead.


Data Handling
-------------

**Q: How do I work with array data?**

A: Use ``ArrayRegisterAccessor``:

   .. code-block:: python

      array = device.getArrayRegisterAccessor("DATA")
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

      array = device.getArrayRegisterAccessor("DATA")
      array.read()

      # Modify
      for i in range(len(array)):
          array[i] = array[i] * 2

      # Write back
      array.write()


**Q: Can I use NumPy with the bindings?**

A: Yes! NumPy is very useful for array operations:

   .. code-block:: python

      import numpy as np

      array = device.getArrayRegisterAccessor("DATA")
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
