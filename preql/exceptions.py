class PreqlError(Exception):
    pass

class PreqlError_Syntax(PreqlError):
    def __str__(self):
        message, context, line, column = self.args
        return '%s at line %s, column %s.\n\n%s' % (message, line, column, context)


class PreqlError_Attribute(PreqlError):
    def __str__(self):
        return "Table '%s' doesn't contain attribute '%s'" % self.args

class PreqlError_MissingName(PreqlError):
    def __str__(self):
        return "Namespace doesn't contain name '%s'" % self.args