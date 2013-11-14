#!/usr/bin/env python
from __future__ import division

import h5py
import numpy as np
from datetime import datetime
from itertools import izip
from scipy.sparse import coo_matrix, csr_matrix, spdiags

class Table(object):
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

        # This call hits us hard performance-wise. Need to either write a
        # smarter/faster indexer or only build index when we need it (lazily).
        self._index_ids()

    def _index_ids(self):
        self._sample_index = self._index_list(self.SampleIds)
        self._obs_index = self._index_list(self.ObservationIds)

    def _index_list(self, l):
        return dict([(id_,idx) for idx,id_ in enumerate(l)])

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
        return self.__class__(self._data.transpose(), self.SampleIds.copy(),
                              self.ObservationIds.copy(), self.TableId)

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

    def copy(self):
        return self.__class__(self._data.copy(), self.ObservationIds.copy(),
                              self.SampleIds.copy(), self.TableId)

    def sum(self, axis='whole'):
        if axis == 'whole':
            axis = None
        elif axis == 'sample':
            axis = 0
        elif axis == 'observation':
            axis = 1
        else:
            raise ValueError

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

    def iterData(self, axis):
        """SLOW... still very slow even when not converting to dense"""
        if axis == 'sample':
            self._data = self._data.tocsc()

            for i in range(self.NumSamples):
                vec = self._data.getcol(i).T
                dense_vec = vec.toarray()

                if vec.shape == (1, 1):
                    result = dense_vec.reshape(1)
                else:
                    result = np.squeeze(dense_vec)

                yield result
        elif axis == 'observation':
            self._data = self._data.tocsr()

            for i in range(self.NumObservations):
                vec = self._data.getrow(i)
                dense_vec = vec.toarray()

                if vec.shape == (1, 1):
                    result = dense_vec.reshape(1)
                else:
                    result = np.squeeze(dense_vec)

                yield result
        else:
            raise ValueError

    def normalize(self, axis):
        if axis == 'sample':
            sum_axis = 0
        elif axis == 'observation':
            sum_axis = 1
        else:
            raise ValueError

        self._data = self._data.tocsr()

        # Requires scipy >= 0.13.0
        norm_data = self._data.multiply(
                csr_matrix(1. / self._data.sum(sum_axis)))
        return self.__class__(norm_data, self.ObservationIds.copy(),
                              self.SampleIds.copy(), self.TableId)

    def sortById(self, axis, sort_f=sorted):
        if axis == 'sample':
            sort_order = np.array(sort_f(self.SampleIds))
        elif axis == 'observation':
            sort_order = np.array(sort_f(self.ObservationIds))
        else:
            raise ValueError

        return self.reorder(axis, sort_order)

    def reorder(self, axis, order):
        if axis == 'sample':
            self._data = self._data.tocsc()
            order_idxs = [self._sample_index[id_] for id_ in order]
            ordered_data = self._data[:,order_idxs]

            return self.__class__(ordered_data, self.ObservationIds.copy(),
                                  order.copy(), self.TableId)
        elif axis == 'observation':
            self._data = self._data.tocsr()
            order_idxs = [self._obs_index[id_] for id_ in order]
            ordered_data = self._data[order_idxs,:]

            return self.__class__(ordered_data, order.copy(),
                                  self.SampleIds.copy(), self.TableId)
        else:
            raise ValueError

    def toFile(self, fp, generated_by='biom-format', storage_type='sparse'):
        out_f = h5py.File(fp, 'w')

        out_f.attrs['id'] = self.TableId
        out_f.attrs['format'] = "Biological Observation Matrix 2.0.0"
        out_f.attrs['format_url'] = "http://biom-format.org"
        out_f.attrs['type'] = 'foo'
        out_f.attrs['generated_by'] = generated_by
        out_f.attrs['date'] = datetime.now().isoformat()
        out_f.attrs['matrix_type'] = storage_type
        out_f.attrs['matrix_element_type'] = self._data.dtype.name
        out_f.attrs['shape'] = self.shape

        out_f['rows'] = self.ObservationIds
        out_f['columns'] = self.SampleIds

        self._data = self._data.tocoo()
        data_grp = out_f.create_group('data')
        data_grp.create_dataset('rows', data=self._data.row)
        data_grp.create_dataset('columns', data=self._data.col)
        data_grp.create_dataset('values', data=self._data.data)

        out_f.close()
