import importlib
import inspect
import pkgutil

from benchflow import BaseBench

__all__ = []

for finder, module_name, is_pkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f".{module_name}", __name__)
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__:
            if issubclass(obj, BaseBench) and obj is not BaseBench:
                globals()[name] = obj
                __all__.append(name)