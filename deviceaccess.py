from . import deviceaccess

try:
    __all__ = tuple(deviceaccess.__all__)
except Exception:
    __all__ = tuple(n for n in dir(deviceaccess) if not n.startswith("_"))

globals().update({name: getattr(deviceaccess, name) for name in __all__})
