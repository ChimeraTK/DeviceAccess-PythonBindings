Troubleshooting
===============

This guide helps you diagnose and solve common issues with the DeviceAccess Python bindings.


Installation Problems
---------------------

Module Import Fails
~~~~~~~~~~~~~~~~~~~~

**Problem:** ``ModuleNotFoundError: No module named 'deviceaccess'``

**Diagnosis:**

.. code-block:: bash

   # Check if the module is installed
   python -c "import deviceaccess; print(deviceaccess.__file__)"

   # Check Python path
   python -c "import sys; print(sys.path)"

   # Verify installation
   pip show chimeratk-deviceaccess

**Solutions:**

1. **Package not installed**: Reinstall it

   .. code-block:: bash

      pip install --upgrade chimeratk-deviceaccess
      # or from source:
      mkdir build && cd build && cmake .. && make && sudo make install

2. **Wrong Python environment**: Ensure you're using the correct Python

   .. code-block:: bash

      which python
      which python3
      # Activate the correct virtual environment if using one

3. **Installation path issue**: Try installing with user flag

   .. code-block:: bash

      pip install --user chimeratk-deviceaccess


Missing Dependencies
~~~~~~~~~~~~~~~~~~~~

**Problem:** ``ImportError: libDOOCSapi.so: cannot open shared object file``

**Solution:** Install the ChimeraTK DeviceAccess library dependency

.. code-block:: bash

   # On Debian/Ubuntu
   sudo apt-get install libchimeratk-deviceaccess

   # Or build and install from source
   git clone https://github.com/ChimeraTK/ChimeraTK-DeviceAccess.git
   cd ChimeraTK-DeviceAccess
   mkdir build && cd build && cmake .. && make && sudo make install


Device Connection Issues
------------------------

Cannot Open Device
~~~~~~~~~~~~~~~~~~~

**Problem:** ``DoocsException: Cannot open device 'MY_DEVICE'``

**Diagnosis Steps:**

1. Check the device map file exists and is accessible:

   .. code-block:: bash

      test -f $DEVICE_MAP_FILE && echo "Map file found" || echo "Map file not found"

2. Verify the device map file syntax:

   .. code-block:: bash

      cat $DEVICE_MAP_FILE
      # Look for proper format: DEVICE_NAME (backend_spec)

3. Test device name is correct:

   .. code-block:: python

      import deviceaccess
      try:
          device = deviceaccess.Device("MY_DEVICE")
          print("Device opened successfully")
      except Exception as e:
          print(f"Error: {e}")

4. Check environment variables:

   .. code-block:: bash

      env | grep -i device
      env | grep -i map

**Solutions:**

1. **Device map file not set**: Ensure the environment variable is set

   .. code-block:: bash

      export DEVICE_MAP_FILE=/path/to/devices.dmap
      python your_script.py

2. **Wrong device name**: Verify the name matches exactly (case-sensitive)

   .. code-block:: python

      # In device map: MY_DEVICE
      # In code: must also be MY_DEVICE (not my_device)
      device = deviceaccess.Device("MY_DEVICE")

3. **Device map syntax error**: Check format is correct

   .. code-block:: text

      # Good: DEVICE_NAME    (backend://spec)
      # Bad: DEVICE_NAME (backend://spec  # Missing space
      # Bad: DEVICE_NAME backend://spec   # Missing parentheses
      TEST (dummy_name_prefix:?)

4. **Backend not available**: The specified backend might not be built

   .. code-block:: bash

      # Check available backends
      apt search chimeratk-device | grep backend
      # or check build logs


Connection Timeout
~~~~~~~~~~~~~~~~~~

**Problem:** ``TimeoutException: Read timeout after waiting X seconds``

**Possible Causes:**

- Device is offline or unreachable
- Network issues (for remote devices)
- Device is busy or locked
- Firewall blocking communication

**Solutions:**

1. **Verify device is online**:

   .. code-block:: bash

      # For network devices, try ping
      ping 192.168.1.100

      # For local devices, check if they're visible
      lsusb  # for USB devices
      dmesg  # for device messages

2. **Check network connectivity**:

   .. code-block:: bash

      # Check firewall rules
      sudo iptables -L -n

      # Try connecting to device port
      telnet 192.168.1.100 502  # For Modbus TCP

3. **Increase timeout** (if supported):

   .. code-block:: python

      # This depends on the backend implementation
      # See backend documentation for timeout options

4. **Enable debug logging** to see what's happening:

   .. code-block:: bash

      export DEVICEACCESS_DEBUG=1
      python your_script.py


Permission Denied
~~~~~~~~~~~~~~~~~

**Problem:** ``DoocsException: Permission denied`` or ``OSError: Permission denied``

**Solutions:**

1. **Insufficient user privileges**: Run with appropriate permissions

   .. code-block:: bash

      # For USB devices, you might need to be in the right group
      sudo usermod -a -G plugdev $USER

      # Or use sudo if necessary
      sudo python your_script.py

2. **Serial port access**: For serial devices

   .. code-block:: bash

      # Add user to dialout group
      sudo usermod -a -G dialout $USER

      # Check device permissions
      ls -l /dev/ttyUSB0

3. **Device file permissions**: If using a socket or special device

   .. code-block:: bash

      # Check who owns the device
      ls -l /path/to/device

      # Give your user access if needed
      sudo chown $USER:$USER /path/to/device


Register Access Issues
----------------------

Register Not Found
~~~~~~~~~~~~~~~~~~~

**Problem:** ``DoocsException: Register 'REGISTER_NAME' not found``

**Diagnosis:**

1. Check the register name is spelled correctly:

   .. code-block:: python

      # In device config: TEMPERATURE_SENSOR
      # In code: must match exactly
      register = device.getScalarRegisterAccessor("TEMPERATURE_SENSOR")

2. Verify the register exists on the device:

   .. code-block:: bash

      # Check device documentation
      # Try with a known register to verify device connection works

**Solutions:**

1. **Spelling error**: Register names are case-sensitive

   .. code-block:: python

      # Wrong: TEMPERATURE (device has TEMPERATURE_SENSOR)
      # Right:
      register = device.getScalarRegisterAccessor("TEMPERATURE_SENSOR")

2. **Wrong device**: Accessing from incorrect device

   .. code-block:: python

      # Check you're using the right device
      print(f"Device opened, looking for: REGISTER_NAME")
      register = device.getScalarRegisterAccessor("REGISTER_NAME")

3. **Backend doesn't expose register**: Some backends filter registers

   .. code-block:: bash

      # Verify register is available via backend
      # Check backend configuration files or device documentation


Type Conversion Errors
~~~~~~~~~~~~~~~~~~~~~~

**Problem:** ``ValueError: Cannot convert to type X``

**Example:**

.. code-block:: python

   register.read()
   value = int(register)  # Fails if register contains text


**Solutions:**

1. **Use correct type**: Match the hardware type

   .. code-block:: python

      # If register is a string
      value = str(register)

      # If register is float
      value = float(register)

2. **Check hardware documentation**: Verify what type the register actually is

3. **Explicit type handling**:

   .. code-block:: python

      import deviceaccess

      try:
          register.read()
          value = float(register)
      except ValueError:
          # Fallback to string
          value = str(register)


Data Inconsistency
~~~~~~~~~~~~~~~~~~

**Problem:** Multiple register values don't seem to match expectations

**Solution:** Use transfer groups for consistency

.. code-block:: python

   # Wrong: Values might be inconsistent
   voltage = device.getScalarRegisterAccessor("VOLTAGE")
   current = device.getScalarRegisterAccessor("CURRENT")

   voltage.read()
   current.read()  # Voltage might have changed between reads

   # Right: Consistent read
   group = device.getTransferGroup()
   group.addAccessor(voltage)
   group.addAccessor(current)
   group.read()  # Both read at same time


Performance Issues
------------------

Read/Write Operations Slow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Diagnosis:**

1. **Are you creating accessors repeatedly?**

   .. code-block:: python

      # Bad: Create accessor in loop
      for i in range(1000):
          register = device.getScalarRegisterAccessor("VALUE")  # Slow!
          register.read()

      # Good: Reuse accessor
      register = device.getScalarRegisterAccessor("VALUE")
      for i in range(1000):
          register.read()

2. **Are you making unnecessary hardware calls?**

   .. code-block:: python

      # Bad: Multiple reads for same data
      register.read()
      for i in range(100):
          register.read()  # Unnecessary!

      # Good: Read once
      register.read()
      value = float(register)
      for i in range(100):
          # Use value, don't read again

3. **Using individual reads instead of transfer groups?**

   .. code-block:: python

      # Bad: N hardware operations
      for reg in registers:
          reg.read()

      # Good: One hardware operation
      group = device.getTransferGroup()
      for reg in registers:
          group.addAccessor(reg)
      group.read()

**Solutions:**

- Reuse accessors
- Minimize hardware access operations
- Use transfer groups
- Cache data between reads if appropriate


Memory Issues
~~~~~~~~~~~~~

**Problem:** High memory usage with large arrays

**Solutions:**

1. **Process arrays in chunks**:

   .. code-block:: python

      import numpy as np

      array = device.getArrayRegisterAccessor("LARGE_ARRAY")
      array.read()

      # Process in chunks
      chunk_size = 1000
      for i in range(0, len(array), chunk_size):
          chunk = np.array(array[i:i+chunk_size])
          # Process chunk...

2. **Use iterators instead of copying**:

   .. code-block:: python

      array = device.getArrayRegisterAccessor("DATA")
      array.read()

      # Iterator doesn't copy all data
      for value in array[:]:
          process(value)


Debugging Tips
--------------

Enable Debug Output
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export DEVICEACCESS_DEBUG=1
   export DEVICE_MAP_DEBUG=1
   python your_script.py


Print Diagnostic Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import deviceaccess
   import sys

   print("Python:", sys.version)
   print("DeviceAccess version:", deviceaccess.__version__)

   # Try opening a device
   try:
       device = deviceaccess.Device("TEST")
       print("Device opened successfully")
   except Exception as e:
       print(f"Error: {e}")
       import traceback
       traceback.print_exc()


Check System Resources
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Check available memory
   free -h

   # Check file descriptors
   ulimit -n

   # Monitor processes
   ps aux | grep python


Use Python Debugger
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pdb
   import deviceaccess

   pdb.set_trace()  # Execution stops here

   # Commands: c (continue), n (next), s (step), p var (print variable)
   device = deviceaccess.Device("MY_DEVICE")


Getting Help
------------

When reporting issues, include:

1. Python version: ``python --version``
2. Library version: ``pip show chimeratk-deviceaccess``
3. OS and platform: ``uname -a``
4. Complete error message and traceback
5. Device map file (sanitized)
6. Minimal reproducible example
7. Debug output (with ``DEVICEACCESS_DEBUG=1``)

**Report to:**

- GitHub Issues: https://github.com/ChimeraTK/ChimeraTK-DeviceAccess-PythonBindings/issues
- ChimeraTK Wiki: https://chimeratk.github.io/
- Your local project documentation


See Also
--------

- :doc:`faq` for common questions
- :doc:`getting_started` for setup help
- :doc:`user_guide` for usage guidance
- API reference for documentation
