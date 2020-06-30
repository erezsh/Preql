import re
import time
import logging

from preql import Preql
from preql.autocomplete import autocomplete
from preql.loggers import test_log

from .common import PreqlTests, SQLITE_URI, POSTGRES_URI

class AutocompleteTests(PreqlTests):
    uri = SQLITE_URI
    optimized = True

    def test_basic(self):
        p = self.Preql()
        state = p.interp.state

        assert "value" in autocomplete(state, "func d(){ [1]{")
        assert "value" in autocomplete(state, "func d(){ [1][")
        assert "value" not in autocomplete(state, "func d(){ [1]")

        res = autocomplete(state, """
        func x(param1) {
            hello = "b"
        """)
        assert "hello" in res, res.keys()

        res = autocomplete(state, """
        func x(param1) {
            hello = "b
        """)

        res = autocomplete(state, """
        func x(param1) {
            hello = [1] {value, value+2}
        """)
        assert "hello" in res, res.keys()

    def test_progressive1(self):
        p = self.Preql()
        state = p.interp.state

        s0 = """
        func hello() = 0

        a = <<<hello>>>
        """
        progressive_test(state, s0)
        progressive_test(state, s0, True)

    def test_progressive2(self):
        p = self.Preql()
        state = p.interp.state

        s1 = """
        func get_users(logins) {
            const table matched_logins = <<<leftjoin>>>(l:logins.value, u:User.login)

            existing_users = <<<matched_logins>>>[<<<u>>>!=null] {<<<u>>>.id}
            new_users = new[] User(login: <<<matched_logins>>>[<<<u>>>==null] {<<<l>>>.value})

            return <<<existing_users>>> + <<<new_users>>>
        }

        hello = <<<get_users>>>([1,2,3])
        do_whatever = <<<hello>>>

        """
        progressive_test(state, s1*2)
        progressive_test(state, s1, True)

    def test_params(self):
        p = self.Preql()
        state = p.interp.state

        s = """
        func enum2(tbl, whatever) = <<<tbl>>> + <<<whatever>>>
        a = <<<enum2>>>
        """
        progressive_test(state, s)

    def test_attr(self):
        p = self.Preql()
        state = p.interp.state

        s = """
        table Country {name: string}

        c = <<<Country>>>
        c = f(<<<Country>>>)
        a = join(c: <<<Country>>>.<<<name>>>, n:["Palau", "Nauru"].<<<value>>>) {c.<<<id>>>, c.<<<name>>>}
        """
        s ="""
        table Country {name: string}
        a = join(c: Country.<<<name>>>, n:["Palau", "Nauru"].<<<value>>>) {n.<<<value>>> => c.<<<name>>>}
        """
        progressive_test(state, s)






def _parse_autocomplete_requirements(s):
    matches = {}
    offset = 0
    def g(m):
        nonlocal offset
        start = m.start() + offset
        x ,= m.groups()
        matches[start] = x
        offset -= 6
        return x

    new_s = re.sub("<<<(\w+)>>>", g, s)
    for k, v in matches.items():
        assert new_s[k:k+len(v)] == v, (k, v)
    return new_s, matches


def progressive_test(state, s, test_partial=False):
    total = 1
    start = time.time()

    s,d = _parse_autocomplete_requirements(s)
    for i in range(1, len(s)):
        ps = s[:i]
        if i in d or test_partial:
            names = autocomplete(state, ps)
            total += 1
            if i in d:
                assert d[i] in names, (i, d[i])

    duration = time.time() - start
    test_log.info(f"Total {total} autocompletions in {duration:.2f} seconds, or {1000*duration/total:.2f} ms per autocomplete")
