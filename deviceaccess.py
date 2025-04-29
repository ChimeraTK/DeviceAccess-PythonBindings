# SPDX-FileCopyrightText: Deutsches Elektronen-Synchrotron DESY, MSK, ChimeraTK Project <chimeratk-support@desy.de>
# SPDX-License-Identifier: LGPL-3.0-or-later
"""
This module offers the functionality of the DeviceAccess C++ library for python.

The ChimeraTK DeviceAccess library provides an abstract interface for register
based devices. Registers are identified by a name and usually accessed though
an accessor object. Since this library also allows access to other control
system applications, it can be understood as the client library of the
ChimeraTK framework.

More information on ChimeraTK can be found at the project's
`github.io <https://chimeratk.github.io/>`_.
"""

from _da_python_bindings import *
