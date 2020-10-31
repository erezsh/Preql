import re
import time

from preql.autocomplete import autocomplete
from preql.loggers import test_log

from .common import PreqlTests, SQLITE_URI, POSTGRES_URI

class AutocompleteTests(PreqlTests):
    uri = SQLITE_URI
    optimized = True

    def test_basic(self):
        p = self.Preql()
        state = p.interp.state

        assert "item" in autocomplete(state, "func d(){ [1]{")
        assert "item" in autocomplete(state, "func d(){ [1][")
        assert "item" not in autocomplete(state, "func d(){ [1]")

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
            hello = [1] {item, item+2}
        """)
        assert "hello" in res, res.keys()

        res = autocomplete(state, """a = [1,2,3]{.""")
        assert res == {}

        res = autocomplete(state, """table a""")
        assert all(isinstance(v, tuple) for v in res.values())

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
            const table matched_logins = <<<leftjoin>>>(l:logins.item, u:User.login)

            existing_users = <<<matched_logins>>>[<<<u>>>!=null] {<<<u>>>.id}
            new_users = new[] User(login: <<<matched_logins>>>[<<<u>>>==null] {<<<l>>>.item})

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

    def test_expr(self):
        p = self.Preql()
        state = p.interp.state

        s = """
        table x {
            a: int
            two: int
            three: int
        }
        <<<x>>>{<<<three>>>}
        <<<x>>>{ => min(<<<two>>>), max(<<<three>>>)}
        """
        progressive_test(state, s)

    def test_exclude_columns(self):
        p = self.Preql()
        state = p.interp.state

        s = """
        table x {
            a: int
            two: int
            three: int
        }
        a = <<<x>>>{... !<<<a>>> !<<<two>>>}{<<<three>>>}
        """
        progressive_test(state, s)

    def test_assert(self):
        p = self.Preql()
        state = p.interp.state

        s = """
        hello = 10
        assert <<<hello>>>
        """
        progressive_test(state, s)


    def test_attr(self):
        p = self.Preql()
        state = p.interp.state

        s = """
        table Country {name: string}

        c = <<<Country>>>
        c = f(<<<Country>>>)
        a = join(c: <<<Country>>>.<<<name>>>, n:["Palau", "Nauru"].<<<item>>>) {c.<<<id>>>, c.<<<name>>>}
        """
        s ="""
        table Country {name: string}
        a = join(c: Country.<<<name>>>, n:["Palau", "Nauru"].<<<item>>>) {n.<<<item>>> => c.<<<name>>>}
        """
        progressive_test(state, s)

    def test_range(self):
        p = self.Preql()
        state = p.interp.state

        s = """
        x=[1,2,3,3,10]
        x order {<<<item>>>} [(<<<count>>>(<<<x>>>/~2))..]
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

    new_s = re.sub(r"<<<(\w+)>>>", g, s)
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
                # if d[i] not in names:
                #     breakpoint()
                #     names = autocomplete(state, ps)

                assert d[i] in names, (i, d[i])

    duration = time.time() - start
    test_log.info(f"Total {total} autocompletions in {duration:.2f} seconds, or {1000*duration/total:.2f} ms per autocomplete")
