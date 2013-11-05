#!/usr/bin/env python
from __future__ import division

import sqlite3
import numpy as np

class Table(object):
    """

    Some of the code in this class is taken and modified from other parts of
    the BIOM project. Credit goes to the contributing authors of that code
    where applicable.
    """

    @classmethod
    def fromFile(cls, table_fp):
        conn = sqlite3.connect(table_fp)
        return cls(conn)

    def __init__(self, conn):
        self.conn = conn

    def __del__(self):
        # TODO: should also (or instead) use a context manager to clean up
        self.conn.commit()
        self.conn.close()
        print 'Cleaned up database'

    @property
    def shape(self):
        return self.NumObservations, self.NumSamples

    @property
    def NumObservations(self):
        c = self.conn.cursor()
        return c.execute("SELECT count(*) FROM observation").fetchone()[0]

    @property
    def NumSamples(self):
        c = self.conn.cursor()
        return c.execute("SELECT count(*) FROM sample").fetchone()[0]

    @property
    def ObservationIds(self):
        c = self.conn.cursor()
        return [r[0] for r in c.execute("SELECT name FROM observation")]

    @property
    def SampleIds(self):
        c = self.conn.cursor()
        return [r[0] for r in c.execute("SELECT name FROM sample")]

    def __str__(self):
        return ('BIOM Table with %d observation(s), %d sample(s), and %f '
                'density' % (self.NumObservations, self.NumSamples,
                             self.tableDensity()))

    def isEmpty(self):
        is_empty = False

        if 0 in self.shape:
            is_empty = True

        return is_empty

    @property
    def nnz(self):
        c = self.conn.cursor()
        return c.execute("SELECT count(*) FROM data").fetchone()[0]

    def tableDensity(self):
        density = 0.0

        if not self.isEmpty():
            density = self.nnz / (self.NumObservations * self.NumSamples)

        return density
