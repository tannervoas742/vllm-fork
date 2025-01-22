class DebugDict:
    def __init__(self):
        self._store = {}

    def set(self, key, value):
        self._store[key] = value

    def get(self, key):
        if key not in self._store:
            return None
        return self._store[key]

    def __repr__(self):
        return str(self._store)

debug_store = DebugDict()