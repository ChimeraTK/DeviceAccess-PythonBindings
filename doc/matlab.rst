=====================================
Using the Python bindings with Matlab
=====================================

To use the ChimeraTK Python bindings inside of Matlab, some steps have to be taken:

* It might be necessary to export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
  before running Matlab, depending on the Matlab version
* It is required to export PYTHONPATH=/usr/share/ChimeraTK-DeviceAccess-PythonBindings-|ProjectVersion|:$PYTHONPATH
  due to the way Matlab handles Python



