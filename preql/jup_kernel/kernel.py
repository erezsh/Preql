from ipykernel.kernelbase import Kernel
# import requests

from . import __version__
import preql
from preql.autocomplete import autocomplete
from preql import Preql, Signal
pql = Preql()
pql.set_output_format('html')

class PreqlKernel(Kernel):
    implementation = 'Preql'
    implementation_version = __version__
    language = 'preql'
    language_version = preql.__version__
    language_info = {
        'name': 'Preql',
        'mimetype': 'text/x-pql',
        'file_extension': '.pql',
        'pygments_lexer': 'go',
    }
    banner = "Preql"

    def do_execute(self, code, silent, store_history=True, user_expressions=None,
                   allow_stdin=False):
        if not silent:
            # r = requests.post("http://127.0.0.1:8080/html", code)
            # json = r.json()
            # res = pql(code)

            # Evaluate (Really just compile)
            try:
                res = pql._run_code(code, '<jupyter>')

                # Print
                if res is not None:
                    res = res.repr(pql.interp.state)

                json = {
                    'output': str(res),
                    'success': True
                }
            except Signal as e:
                json = {
                    'output': '<pre>%s</pre>' % str(e),
                    'success': False
                }


            if json['success']:
                stream_content = {'name': 'stdout', 'text': json['output']}
            else:
                stream_content = {'name': 'stderr', 'text': json['output']}
            # self.send_response(self.iopub_socket, 'stream', stream_content)

            html = json['output']
            # self.send_info("Elapsed Time: {} !\n".format(elapsed_time))
            self.send_response(self.iopub_socket, 'display_data', {
                'data': {
                    "text/html": html,
                },
                "metadata": {
                    "image/png": {
                        "width": 640,
                        "height": 480,
                    },
                }
            })


        return {'status': 'ok',
                # The base class increments the execution count
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {},
               }


    def do_complete(self, code, cursor_pos):
        context, fragment = last_word(code[:cursor_pos])

        all_vars = autocomplete(pql.interp.state, context)
        matches = [f'{k}' for k in all_vars if k.startswith(fragment)]

        return {
            'status' : 'ok',
            'matches' : matches,
            'cursor_start' : cursor_pos - len(fragment),
            'cursor_end' : cursor_pos,
            'metadata' : {},
        }


def is_name(s):
    return s.isalnum() or s in ('_', '!')
def last_word(s):
    if not s:
        return '', ''
    i = len(s)
    while i and is_name(s[i-1]):
        i -= 1
    if i < len(s) and s[i] == '!' :
        i += 1  # hack to support ... !var and !in
    return s[:i], s[i:]