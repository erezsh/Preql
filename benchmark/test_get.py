"""
Benchmark for constant-time overhead over Sqlite3

Benchmarks the following:
- Preql (with cached compilation)
- SqlAlchemy
- Raw Sqlite

NOTE: Cached compilation is still an experimental feature!
"""

import time
from preql.utils import benchmark
from preql import Preql, settings
from sqlalchemy import create_engine, Column, Integer
from sqlalchemy.orm import sessionmaker, load_only
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
engine = create_engine('sqlite:///:memory:')

class Vector(Base):
    __tablename__ = 'Vector'
    id = Column(Integer, primary_key=True)
    v = Column(Integer)

VECTOR_COUNT = 1000

settings.typecheck = False
settings.cache = True

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
pql = Preql()
conn = engine.raw_connection().connection
pql._interp.state.db._conn = conn
cur = conn.cursor()
pql('''
    table Vector {
        id: t_id
        ...
    }
    func get(id_) = Vector[id==id_]{v}
''')


def pql_get(id_):
    res = pql.get(id_).to_json()
    if len(res) != 1:
        raise ValueError()


def raw_get(id_):
    cur.execute('SELECT v FROM Vector WHERE id=%s' % id_)
    res = cur.fetchall()
    if len(res) != 1:
        raise ValueError()


def sqlalchemy_get_2(id_):
    res = session.query(Vector).filter_by(id=id_).options(load_only('v'))
    if res.count() != 1:
        raise ValueError()


def sqlalchemy_get(id_):
    res = session.query(Vector).filter_by(id=id_)
    if res.count() != 1:
        raise ValueError()


def run_benchmark(fs, count):
    res = []
    for f in fs:
        print('.. ', f)
        start = time.time()
        for i in range(count):
            f(i % (VECTOR_COUNT//2) + 1)
        else:
            end = time.time()
            res.append((end - start, f))

    else:
        res.sort(key=(lambda x: x[0]))
        top_t = res[0][0] + 1e-07
        for t, f in res:
            print('%.4f\t%s\t(x%.2f)' % (t, f.__name__, t / top_t))
        else:
            benchmark.print()


def main():
    print('Initializing..')
    for i in range(VECTOR_COUNT):
        pql('new Vector(%d)' % i)

    print('Running benchmark')
    benchmark.reset()
    run_benchmark([raw_get, pql_get, sqlalchemy_get, sqlalchemy_get_2], 1000)


main()
