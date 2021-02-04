from docutils import nodes
from docutils.parsers.rst import Directive

from preql.docstring.autodoc import generate_rst
# class HelloWorld(Directive):

#     def run(self):
#         paragraph_node = nodes.paragraph(text='Hello World!')
#         return [paragraph_node]


def setup(app):
    # app.add_directive("helloworld", HelloWorld)
    print("Generating documentation for Preql's modules")
    generate_rst('preql-modules.rst', 'preql-types.rst')

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }