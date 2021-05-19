
from typing import Optional, List, Union

from lark import Lark, Transformer, v_args
from lark.indenter import Indenter

from runtype import dataclass
from rich.markup import escape as rich_esc


def indent_str(indent):
    return ' ' * (indent)

@dataclass
class Text:
    lines: List[str]

    def _print_text(self, indent, inline=False):
        if not self.lines:
            return
        s = indent_str(indent)
        if inline:
            yield self.lines[0] + '\n'
        else:
            yield s + self.lines[0] + '\n'
        for line in self.lines[1:]:
            yield s + line + '\n'

    def _print_html(self):
        for line in self.lines:
            yield line
            yield '<br/>\n'

    def _print_rst(self, section=None):
        yield from self._print_text(8)



@dataclass(frozen=False)
class Defin:
    name: str
    text: Optional[Text]
    type: str = ''
    default: str = ''

    def _print_text(self, indent):
        if self.type:
            type = ' (%s)' % rich_esc(self.type)
        else:
            type = ''
        decl = f'[bold]{self.name}[/bold]{type}'
        yield f'{indent_str(indent)}{decl}: '
        if self.text:
            yield from self.text._print_text(indent+len(decl)+2, True)
        else:
            yield '\n'

    def _print_html(self):
        if self.type:
            type = '<em>(%s)</em>' % self.type
        else:
            type = ''
        yield f'<b>{self.name}</b>{type}'
        yield from self.text._print_html()

    def _print_rst(self):
        text = self.text.replace(lines = self.text.lines + [f'(default={self.default})']) if self.default else self.text
        text = ''.join(text._print_rst()) if self.text else '\n'
        yield f"    :param  {self.name}: {text}"
        if self.type:
            yield f"    :type  {self.name}: {self.type}\n"


@dataclass
class Section:
    name: str
    items: List[Union[Defin, Text, object]]

    def _print_text(self, indent):
        l = [f'[bold white]{indent_str(indent)}{self.name}[/bold white]:\n']
        for item in self.items:
            l += item._print_text(indent+4)
        return l

    def _print_html(self):
        yield '<div class="section">\n'
        yield '<h3>'
        yield self.name + ":"
        yield '</h3>\n'
        for item in self.items:
            yield from item._print_html()
        yield '</div>\n'

    def _print_rst(self):
        if self.name != 'Parameters':
            yield '    :' + self.name + ':\n\n'

        if self.name in ('Example', 'Examples'):
            yield '        .. code-block:: javascript\n\n'
            for item in self.items:
                yield from item._print_text(12)
        else:
            for item in self.items:
                yield from item._print_rst()

@dataclass
class DocString:
    header: Text
    sections: List[Section]

    def get_section(self, name):
        for s in self.sections:
            if s.name == name:
                return s
        raise KeyError(name)

    def _print_text(self, indent):
        yield from self.header._print_text(indent)
        yield '\n'
        for section in self.sections:
            yield from section._print_text(indent)
            yield '\n'

    def _print_html(self):
        yield '<div class="doc">\n'
        for section in self.sections:
            yield from section._print_html()
        yield '</div>\n'

    def _print_rst(self):
        yield from self.header._print_text(4)
        yield '\n'
        for section in self.sections:
            yield from section._print_rst()
            yield '\n'

    def print_text(self, indent=0):
        return ''.join(self._print_text(indent))

    def print_html(self):
        return ''.join(self._print_html())

    def print_rst(self):
        return ''.join(self._print_rst())



_inline = v_args(inline=True)
class DocTransformer(Transformer):
    def as_list(self, items):
        return items

    # attrs = as_list
    section_items = as_list
    sections = as_list

    text = Text
    header = Text
    start = _inline(DocString)

    @_inline
    def section(self, name, items):
        return Section(name.rstrip(':'), items)

    @_inline
    def defin(self, name, text):
        return Defin(name.rstrip(':'), text)



class DocIndenter(Indenter):
    NL_type = '_NL'
    OPEN_PAREN_types = []
    CLOSE_PAREN_types = []
    INDENT_type = '_INDENT'
    DEDENT_type = '_DEDENT'
    tab_len = 1


parser = Lark.open('docstring.lark', rel_to=__file__,
                    parser='lalr', #lexer=_Lexer,
                    postlex=DocIndenter(),
                    maybe_placeholders=True,
                    )


def parse(s):
    s = s.strip()
    if not s:
        return DocString(Text([]), [])
    tree = parser.parse(s+'\n')
    return DocTransformer().transform(tree)


# def test_parser():
#     s ="""
#     Ok

#     Parameters:
#         Test1: bla
#     """

#     """
#     Parameters:
#         Test_1: whatever
#         Param2(int, optional): LALALA
#                             BLA BLA BLA
#                             YESYES
#                                 WHAT NOW???

#     See Also:
#         Whatever
#             OK
#         This counts too

#     """
#     res = parse(s)
#     # print(res)
#     # print(res.print_html())


if __name__ == "__main__":
    test_parser()