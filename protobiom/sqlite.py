#!/usr/bin/env python
from __future__ import division

import sqlite3
import numpy as np

class Table(object):
    @classmethod
    def fromFile(cls, table_fp):
        conn = sqlite3.connect(table_fp)
        return cls(conn)

    def __init__(self, conn):
        self.conn = conn

    def __del__(self):
        # Should also (or instead) use a context manager to clean up.
        self.conn.commit()
        self.conn.close()

    # These simple properties could be lazily-loaded to improve performance.

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

    def sum(self, axis='whole'):
        c = self.conn.cursor()

        if axis == 'whole':
            return c.execute("SELECT sum(abundance) FROM data").fetchone()[0]
        elif axis == 'sample':
            #return [r[0] for r in c.execute("SELECT sum(abundance) FROM data GROUP BY sample_id")]
            return c.execute("SELECT name, sum(abundance) FROM data, sample WHERE data.sample_id = sample.id GROUP BY sample_id").fetchall()
        elif axis == 'observation':
            return c.execute("SELECT name, sum(abundance) FROM data, observation WHERE data.observation_id = observation.id GROUP BY observation_id").fetchall()
        else:
            raise ValueError

    @property
    def nnz(self):
        c = self.conn.cursor()
        return c.execute("SELECT count(*) FROM data").fetchone()[0]

    def tableDensity(self):
        density = 0.0

        if not self.isEmpty():
            density = self.nnz / (self.NumObservations * self.NumSamples)

        return density

    def iterData(self, axis):
        c = self.conn.cursor()

        if axis == 'sample':
            # This is *extremely* slow.
            for i in range(1, self.NumSamples + 1):
                row = np.zeros(self.NumObservations)

                for r in c.execute("SELECT observation_id, abundance FROM data WHERE sample_id = ?", (i,)):
                    row[r[0] - 1] = r[1]

                yield row
        elif axis == 'observation':
            for i in range(1, self.NumObservations + 1):
                row = np.zeros(self.NumSamples)

                for r in c.execute("SELECT sample_id, abundance FROM data WHERE observation_id = ?", (i,)):
                    row[r[0] - 1] = r[1]

                yield row
        else:
            raise ValueError
