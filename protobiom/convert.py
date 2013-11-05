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

class TableReader(object):
    def __init__(self, table_str):
        self.table_str = table_str

    @property
    def table_id(self):
        if not hasattr(self, '_table_id'):
            self._table_id = self._parse_field('id')
        return self._table_id

    @property
    def version(self):
        if not hasattr(self, '_version'):
            self._version = self._parse_field('format')
        return self._version

    @property
    def url(self):
        if not hasattr(self, '_url'):
            self._url = self._parse_field('format_url')
        return self._url

    @property
    def table_type(self):
        if not hasattr(self, '_table_type'):
            self._table_type = self._parse_field('type')
        return self._table_type

    @property
    def generated_by(self):
        if not hasattr(self, '_generated_by'):
            self._generated_by = self._parse_field('generated_by')
        return self._generated_by

    @property
    def date(self):
        if not hasattr(self, '_date'):
            self._date = self._parse_field('date')
        return self._date

    @property
    def matrix_type(self):
        if not hasattr(self, '_matrix_type'):
            self._matrix_type = self._parse_field('matrix_type')
        return self._matrix_type

    @property
    def matrix_element_type(self):
        if not hasattr(self, '_matrix_element_type'):
            self._matrix_element_type = self._parse_field('matrix_element_type')
        return self._matrix_element_type

    @property
    def comment(self):
        if not hasattr(self, '_comment'):
            self._comment = self._parse_field('comment', required=False)
        return self._comment

    @property
    def shape(self):
        if not hasattr(self, '_shape'):
            self._shape = self._parse_shape()
        return self._shape

    @property
    def observation_ids(self):
        if not hasattr(self, '_observation_ids'):
            self._observation_ids = self._parse_metadata('rows')
        return self._observation_ids

    @property
    def sample_ids(self):
        if not hasattr(self, '_sample_ids'):
            self._sample_ids = self._parse_metadata('columns')
        return self._sample_ids

    def data(self):
        search_str = '"data": [['
        start_idx = self.table_str.index(search_str) + len(search_str)

        while True:
            end_idx = self.table_str.index(']', start_idx)
            data_strs = self.table_str[start_idx:end_idx].split(',')
            assert len(data_strs) == 3
            yield int(data_strs[0]), int(data_strs[1]), float(data_strs[2])

            if self.table_str[end_idx + 1] == ',':
                start_idx = end_idx + 3
            else:
                break

    def _parse_field(self, field, required=True):
        search_str = '"%s": "' % field

        try:
            start_idx = self.table_str.index(search_str) + len(search_str)
        except ValueError:
            if required:
                raise ValueError("Missing required field '%s'." % field)
            else:
                return None
        else:
            end_idx = self.table_str.index('",', start_idx)
            return self.table_str[start_idx:end_idx]

    def _parse_shape(self):
        search_str = '"shape": ['
        start_idx = self.table_str.index(search_str) + len(search_str)
        end_idx = self.table_str.index('],', start_idx)
        dim_strs = self.table_str[start_idx:end_idx].split(', ')
        assert len(dim_strs) == 2
        return tuple(map(int, dim_strs))

    def _parse_metadata(self, axis):
        search_str = '"%s": [{' % axis
        start_idx = self.table_str.index(search_str) + len(search_str)

        md = []
        while True:
            end_idx = self.table_str.index('}', start_idx)
            fields = self.table_str[start_idx:end_idx].split(', ')
            assert len(fields) == 2
            md.append(fields[0].split('"id": "')[1][:-1])

            if self.table_str[end_idx + 1] == ',':
                start_idx = end_idx + 3
            else:
                break
        return md
