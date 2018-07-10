

class OntologyRegistry:
    """
    Contains all classes usable for constructing games
    """
    def __init__(self):
        self._register = {}


    def __contains__(self, key):
        return key in self._register


    def register(self, key, cls):
        self._register[key] = cls


    def register_class(self, cls):
        return self.register(cls.__class__.__name__, cls)


    def request(self, key):
        return self._register[key]


    def register_all(self, module):
        import inspect

        for key, obj in inspect.getmembers(module):
            if key.startswith('__'):
                continue
            # obj can be anything, class or function,...
            self.register(key, obj)


registry = OntologyRegistry()
