
class Object:
    def repr(self):
        return repr(self)

    def inline_repr(self):
    	return self.repr()

    def all_attrs(self):
        return {}

    get_attr = NotImplemented
    isa = NotImplemented
