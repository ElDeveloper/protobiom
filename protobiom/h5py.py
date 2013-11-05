#!/usr/bin/env python
from __future__ import division

import h5py
import numpy as np
from itertools import izip
from scipy.sparse import coo_matrix, csr_matrix, spdiags

class Table(object):
    """

    Some of the code in this class is taken and modified from other parts of
    the BIOM project. Credit goes to the contributing authors of that code
    where applicable.
    """

    @classmethod
    def fromFile(cls, table_fp):
        table_f = h5py.File(table_fp, 'r')
        obs_ids = table_f['rows'].value
        sample_ids = table_f['columns'].value
        table_id = table_f.attrs['id']

        data_grp = table_f['data']
        matrix = coo_matrix((data_grp['values'].value,
                            (data_grp['rows'].value,
                             data_grp['columns'].value)),
                            shape=table_f.attrs['shape'])
        table_f.close()

        return cls(matrix, obs_ids, sample_ids, table_id)

    def __init__(self, data, ObservationIds, SampleIds, TableId=None):
        self._data = data
        self.ObservationIds = ObservationIds
        self.SampleIds = SampleIds
        self.TableId = TableId

    @property
    def shape(self):
        return self._data.shape

    @property
    def NumObservations(self):
        return self.shape[0]

    @property
    def NumSamples(self):
        return self.shape[1]

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
        return self.__class__(self._data.transpose(), self.SampleIds[:],
                              self.ObservationIds[:], self.TableId)

    def __eq__(self, other):
        eq = True

        if not isinstance(other, self.__class__):
            eq = False
        if self.shape != other.shape:
            eq = False
        if (self.ObservationIds != other.ObservationIds).any():
            eq = False
        if (self.SampleIds != other.SampleIds).any():
            eq = False
        if not self.isEmpty():
            # Requires scipy >= 0.13.0
            if (self._data != other._data).nnz > 0:
                eq = False

        return eq

    def __ne__(self, other):
        return not (self == other)

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
        if axis == 'whole':
            axis = None
        elif axis == 'sample':
            axis = 0
        elif axis == 'observation':
            axis = 1
        else:
            raise ValueError("Unrecognized axis '%s'" % axis)

        matrix_sum = np.squeeze(np.asarray(self._data.sum(axis=axis)))

        if axis is not None and matrix_sum.shape == ():
            matrix_sum = matrix_sum.reshape(1)

        return matrix_sum

    def tableDensity(self):
        density = 0.0

        if not self.isEmpty():
            density = self._data.nnz / (self.shape[0] * self.shape[1])

        return density

    def nonzero(self):
        return self._data.nonzero()

    def iterObservationData(self):
        """SLOW... still very slow even when not converting to dense"""
        self._data = self._data.tocsr()

        for i in range(self.NumObservations):
            vec = self._data.getrow(i)
            dense_vec = np.asarray(vec.todense())

            if vec.shape == (1, 1):
                result = dense_vec.reshape(1)
            else:
                result = np.squeeze(dense_vec)

            yield result
            yield vec

    # For sorting, row/col swapping detailed here may be useful:
    # http://stackoverflow.com/questions/15155276/rearrange-sparse-arrays-by-swapping-rows-and-columns

    def normalize(self, axis):
        # Another solution: http://stackoverflow.com/a/12238133

        # Could also use sklearn: http://stackoverflow.com/a/12396922
        #import sklearn.preprocessing
        #return sklearn.preprocessing.normalize(self._data, axis=1, norm='l1')

        # From http://stackoverflow.com/a/8359856
        # Only seems to work for square matrices though...
        #ccd = spdiags(1./self._data.sum(1).T, 0, *self._data.shape)
        #ccn = ccd * self._data.T
        #return ccn

        if axis == 'sample':
            sum_axis = 0
        elif axis == 'observation':
            sum_axis = 1
        else:
            raise ValueError("Unrecognized axis '%s'" % axis)

        self._data = self._data.tocsr()

        # Requires scipy >= 0.13.0
        norm_data = self._data.multiply(
                csr_matrix(1. / self._data.sum(sum_axis)))
        return self.__class__(norm_data, self.ObservationIds[:],
                              self.SampleIds[:], self.TableId)
