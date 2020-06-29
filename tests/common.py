from unittest import TestCase, skip
from preql import sql, settings

from preql import Preql

SQLITE_URI = 'sqlite://:memory:'
POSTGRES_URI = 'postgres://postgres:qweqwe123@localhost/postgres'

class PreqlTests(TestCase):
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