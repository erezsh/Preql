
class Object:
    def repr(self):
        return repr(self)

    def all_attrs(self):
        return {}

    get_attr = NotImplemented
    isa = NotImplemented
