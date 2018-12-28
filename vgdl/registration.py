from typing import Union
from types import ModuleType


class OntologyRegistry:
    """
    Contains all classes usable for constructing games
    """

    def __init__(self):
        self._register = {}

    def __contains__(self, key):
        return key in self._register

    def register(self, key, cls):
        if key in self._register:
            # re-registering can have funky behaviour when classes are reloaded
            # raise KeyError('`{}` already registered'.format(cls.__name__))
            pass
        self._register[key] = cls

    def register_class(self, cls):
        return self.register(cls.__name__, cls)

    def request(self, key):
        return self._register[key]

    def register_all(self, module: ModuleType):
        # Had an issue with re-registering classes that were imported in module
        if isinstance(module, ModuleType):
            import inspect

            # Respect a module's exports
            module_all = module.__all__ if hasattr(module, '__all__') else None

            for key, obj in inspect.getmembers(module):
                if key.startswith('__'):
                    continue
                if module_all and key not in module_all:
                    continue
                # obj can be anything, class or function,...
                self.register(key, obj)
        else:
            raise TypeError('Not sure how to register %s of type %s' % (module, type(module)))

    def register_from_string(self, module: str):
        """ module is expected to be a dot-separated Python module spec """
        import importlib
        module = importlib.import_module(module)
        self.register_all(module)


registry = OntologyRegistry()
