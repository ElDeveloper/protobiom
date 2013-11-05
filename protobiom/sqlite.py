#!/usr/bin/env python

#-----------------------------------------------------------------------------
# Copyright (c) 2011-2013, The BIOM Format Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from __future__ import division

__author__ = "Jai Ram Rideout"
__copyright__ = "Copyright 2011-2013, The BIOM Format Development Team"
__credits__ = ["Jai Ram Rideout"]
__license__ = "BSD"
__url__ = "http://biom-format.org"
__version__ = "1.2.0-dev"
__maintainer__ = "Jai Ram Rideout"
__email__ = "jai.rideout@gmail.com"

import sqlite3
from itertools import izip
import numpy as np
from biom.exception import TableException

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
