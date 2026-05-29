Troubleshooting
===============

This guide helps you diagnose and solve common issues with the DeviceAccess Python bindings.

Performance Issues
------------------

Read/Write Operations Slow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Diagnosis:**

1. **Are you creating accessors repeatedly?**

   .. code-block:: python

      # Bad: Create accessor in loop
      for i in range(1000):
          register = device.getScalarRegisterAccessor(int, "VALUE")  # Slow!
          register.read()

      # Good: Reuse accessor
      register = device.getScalarRegisterAccessor(int,"VALUE")
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


**Solutions:**

- Reuse accessors
- Minimize hardware access operations
- Cache data between reads if appropriate


Getting Help
------------

When reporting issues, include:

1. Python version: ``python --version``
2. Library version: ``apt show python3-mtca4upy``
3. OS and platform: ``uname -a``
4. Complete error message and traceback
5. Device map file (sanitized)
6. Minimal reproducible example

**Report to:**

- Inside DESY: Redmine
- GitHub Issues: https://github.com/ChimeraTK/ChimeraTK-DeviceAccess-PythonBindings/issues


See Also
--------

- :doc:`faq` for common questions
- :doc:`getting_started` for setup help
- :doc:`user_guide` for usage guidance
- API reference for documentation
