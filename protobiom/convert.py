#!/usr/bin/env python
from __future__ import division
import json

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
            self._observation_ids = self._parse_ids('rows')
        return self._observation_ids

    @property
    def sample_ids(self):
        if not hasattr(self, '_sample_ids'):
            self._sample_ids = self._parse_ids('columns')
        return self._sample_ids

    @property
    def observation_metadata(self):
        if not hasattr(self, '_observation_metadata'):
            self._observation_metadata = self._parse_metadata('rows')
        return self._observation_metadata

    @property
    def sample_metadata(self):
        if not hasattr(self, '_sample_metadata'):
            self._sample_metadata = self._parse_metadata('columns')
        return self._sample_metadata

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

    def _parse_ids(self, axis):
        search_str = '"%s": [{' % axis
        start_idx = self.table_str.index(search_str) + len(search_str) - 2
        end_idx = self.table_str.index('}]', start_idx) + 2
        md_str = self.table_str[start_idx:end_idx]

        ids = []
        for e in json.loads(md_str):
            ids.append(str(e['id']))

        return ids

    def _parse_metadata(self, axis):
        search_str = '"%s": [{' % axis
        start_idx = self.table_str.index(search_str) + len(search_str) - 2
        end_idx = self.table_str.index('}]', start_idx) + 2
        md_str = self.table_str[start_idx:end_idx]

        md = []
        for e in json.loads(md_str):
            e_md = e['metadata']

            if e_md is None:
                return None
            else:
                md.append(str(';'.join(e['metadata']['taxonomy'])))

        return md
