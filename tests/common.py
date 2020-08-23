from unittest import TestCase, skip
from preql import sql, settings

from preql import Preql

SQLITE_URI = 'sqlite://:memory:'
POSTGRES_URI = 'postgres://postgres:qweqwe123@localhost/postgres'
MYSQL_URI = 'mysql://erez:qweqwe123@localhost/preql_tests'

class PreqlTests(TestCase):
    optimized = True
    uri = SQLITE_URI

    def Preql(self):
        settings.optimize = self.optimized
        preql = Preql(self.uri)
        self.preql = preql
        return preql

    def setUp(self):
        self.preql = None

    def tearDown(self):
        if self.preql:
            self.preql.engine.rollback()