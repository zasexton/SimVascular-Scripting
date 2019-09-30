import weakref
class node:
    _instances = set()
    def __init__(self,coordinates,unique_id):
        self.coordinates = coordinates
        self.id = unique_id
        self._instances.add(weakref.ref(self))
    @classmethod
    def getinstances(cls):
        ref_return = set()
        for pointer in cls._instances:
            obj = pointer()
            if obj is not None:
                yield obj
            else:
                ref_return.add(pointer)
        cls._instances -= ref_return
