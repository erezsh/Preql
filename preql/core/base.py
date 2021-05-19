
class Object:
    def repr(self):
        return repr(self)

    def inline_repr(self):
    	return self.repr()

    def rich_repr(self):
        return self.repr().replace('[', '\\[')


    def all_attrs(self):
        return {}

    get_attr = NotImplemented
    isa = NotImplemented
