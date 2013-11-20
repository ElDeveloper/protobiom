#!/usr/bin/env python
from __future__ import division

import h5py
import numpy as np
from datetime import datetime
from itertools import izip
from scipy.sparse import coo_matrix, csr_matrix

class Table(object):
    @classmethod
    def fromFile(cls, table_fp):
        table_f = h5py.File(table_fp, 'r')

        obs_ids = table_f['observations/ids'].value
        if 'observations/metadata' in table_f:
            obs_md = table_f['observations/metadata'].value

        sample_ids = table_f['samples/ids'].value
        if 'samples/metadata' in table_f:
            sample_md = table_f['samples/metadata'].value

        table_id = table_f.attrs['id']
        table_type = table_f.attrs['type']

        data_grp = table_f['data']
        matrix = coo_matrix((data_grp['values'].value,
                            (data_grp['rows'].value,
                             data_grp['columns'].value)),
                            shape=table_f.attrs['shape'])
        table_f.close()

        return cls(matrix, obs_ids, sample_ids, ObservationMetadata=obs_md,
                   TableId=table_id, TableType=table_type)

    def __init__(self, data, ObservationIds, SampleIds,
                 ObservationMetadata=None, SampleMetadata=None, TableId=None,
                 TableType=None):
        self._data = data
        self.ObservationIds = ObservationIds
        self.SampleIds = SampleIds

        if ObservationMetadata is None:
            ObservationMetadata = np.array([None] * len(self.ObservationIds))
        self.ObservationMetadata = ObservationMetadata

        if SampleMetadata is None:
            SampleMetadata = np.array([None] * len(self.SampleIds))
        self.SampleMetadata = SampleMetadata
        self.TableId = TableId
        self.TableType = TableType

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
        self._data = self._data.transpose()

        obs_ids = self.ObservationIds
        self.ObservationIds = self.SampleIds
        self.SampleIds = obs_ids

        obs_md = self.ObservationMetadata
        self.ObservationMetadata = self.SampleMetadata
        self.SampleMetadata = obs_md

        self._invalidate_indices('observation')
        self._invalidate_indices('sample')

        return self

    def __eq__(self, other):
        eq = True

        if not isinstance(other, self.__class__):
            eq = False
        elif self.shape != other.shape:
            eq = False
        elif (self.ObservationIds != other.ObservationIds).any():
            eq = False
        elif (self.SampleIds != other.SampleIds).any():
            eq = False
        elif not self.isEmpty():
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

    def hasMetadata(self, axis):
        if axis == 'observation':
            md = self.ObservationMetadata
        elif axis == 'sample':
            md = self.SampleMetadata
        else:
            raise ValueError

        return not np.equal(md, None).all()

    def copy(self):
        return self.__class__(self._data.copy(), self.ObservationIds.copy(),
                              self.SampleIds.copy(),
                              self.ObservationMetadata.copy(),
                              self.SampleMetadata.copy(), self.TableId,
                              self.TableType)

    def sum(self, axis='whole'):
        if axis == 'whole':
            axis = None
        elif axis == 'observation':
            axis = 1
        elif axis == 'sample':
            axis = 0
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
        """
        
        WARNING: very slow! Still very slow even when not converting to dense.
        """
        if axis == 'observation':
            self._data = self._data.tocsr()

            for i in range(self.NumObservations):
                vec = self._data.getrow(i)
                dense_vec = vec.toarray()

                if vec.shape == (1, 1):
                    result = dense_vec.reshape(1)
                else:
                    result = np.squeeze(dense_vec)

                yield result
        elif axis == 'sample':
            self._data = self._data.tocsc()

            for i in range(self.NumSamples):
                vec = self._data.getcol(i).T
                dense_vec = vec.toarray()

                if vec.shape == (1, 1):
                    result = dense_vec.reshape(1)
                else:
                    result = np.squeeze(dense_vec)

                yield result
        else:
            raise ValueError

    def normalize(self, axis):
        if axis == 'observation':
            sum_axis = 1
        elif axis == 'sample':
            sum_axis = 0
        else:
            raise ValueError

        self._data = self._data.tocsr()

        # Requires scipy >= 0.13.0
        self._data = self._data.multiply(
                csr_matrix(1. / self._data.sum(sum_axis)))

        return self

    def sortById(self, axis, sort_f=np.sort):
        if axis == 'observation':
            sort_order = sort_f(self.ObservationIds)
        elif axis == 'sample':
            sort_order = sort_f(self.SampleIds)
        else:
            raise ValueError

        return self.reorder(axis, sort_order)

    def reorder(self, axis, order):
        if axis == 'observation':
            self._data = self._data.tocsr()
            order_idxs = [self.index('observation', id_) for id_ in order]
            self._data = self._data[order_idxs,:]
            self.ObservationIds = self.ObservationIds[order_idxs]
            self.ObservationMetadata = self.ObservationMetadata[order_idxs]
            self._invalidate_indices('observation')
        elif axis == 'sample':
            self._data = self._data.tocsc()
            order_idxs = [self.index('sample', id_) for id_ in order]
            self._data = self._data[:,order_idxs]
            self.SampleIds = self.SampleIds[order_idxs]
            self.SampleMetadata = self.SampleMetadata[order_idxs]
            self._invalidate_indices('sample')
        else:
            raise ValueError

        return self

    def index(self, axis, id_):
        if axis == 'observation':
            if not hasattr(self, '_obs_index'):
                self._obs_index = self._index_list(self.ObservationIds)
            return self._obs_index[id_]
        elif axis == 'sample':
            if not hasattr(self, '_sample_index'):
                self._sample_index = self._index_list(self.SampleIds)
            return self._sample_index[id_]
        else:
            raise ValueError

    def _invalidate_indices(self, axis):
        if axis == 'observation':
            if hasattr(self, '_obs_index'):
                del self._obs_index
        elif axis == 'sample':
            if hasattr(self, '_sample_index'):
                del self._sample_index
        else:
            raise ValueError

    def _index_list(self, l):
        return dict([(id_, idx) for idx, id_ in enumerate(l)])

    def toFile(self, fp, generated_by='biom-format'):
        out_f = h5py.File(fp, 'w')

        out_f.attrs['id'] = self.TableId
        out_f.attrs['format'] = "Biological Observation Matrix 2.0.0"
        out_f.attrs['format_url'] = "http://biom-format.org"
        out_f.attrs['type'] = self.TableType
        out_f.attrs['generated_by'] = generated_by
        out_f.attrs['date'] = datetime.now().isoformat()
        out_f.attrs['matrix_element_type'] = self._data.dtype.name
        out_f.attrs['shape'] = self.shape

        obs_grp = out_f.create_group('observations')
        obs_grp['ids'] = self.ObservationIds
        if self.hasMetadata('observation'):
            obs_grp['metadata'] = self.ObservationMetadata

        samp_grp = out_f.create_group('samples')
        samp_grp['ids'] = self.SampleIds
        if self.hasMetadata('sample'):
            samp_grp['metadata'] = self.SampleMetadata

        self._data = self._data.tocoo()
        data_grp = out_f.create_group('data')
        data_grp.create_dataset('rows', data=self._data.row)
        data_grp.create_dataset('columns', data=self._data.col)
        data_grp.create_dataset('values', data=self._data.data)

        out_f.close()
