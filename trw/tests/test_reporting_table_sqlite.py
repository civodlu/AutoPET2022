import copy
import os
import tempfile

import trw
from unittest import TestCase
import sqlite3
import numpy as np
from trw.reporting import TableStream, export_sample
from trw.utils import len_batch
from trw.reporting.table_sqlite import get_tables_name_and_role, get_metadata_name, get_table_data


def get_table_values(connection, table_name):
    cursor = connection.execute(f'select * from {table_name}')
    rows = cursor.fetchall()
    return rows


class TestReportingTableSqlite(TestCase):
    def test_table_basics(self):
        # make sure we can create tables, insert and retrieve the data
        # as expected
        connection = sqlite3.connect(':memory:')
        cursor = connection.cursor()

        batch = {
            'split': ['train'] * 3,
            'epoch': [0, 0, 0],
            'value1': [1, 2, 3],
            'strings': ['p1', 'p2', 'p3']
        }

        table = trw.reporting.TableStream(cursor, 'table_test', 'role_test', table_preamble='table_preamble_value')

        # this will actually create the table
        table.insert(batch)

        # insert without table creation
        table.insert(batch)

        # make sure the 2 tables are created
        sql_command = f"SELECT name FROM sqlite_master WHERE type = 'table';"
        cursor.execute(sql_command)
        rows = cursor.fetchall()
        assert len(rows) == 2
        row_names = {n for n, in rows}

        assert 'table_test' in row_names
        assert 'table_test_metadata' in row_names

        # make sure we have the expected rows
        d = table.get_column_names()
        for key in batch.keys():
            assert key in d

        values = get_table_values(connection, 'table_test')
        assert len(values) == 6

        data_roles = get_tables_name_and_role(connection)
        assert len(data_roles) == 1
        assert len(data_roles[0]) == 2
        assert data_roles[0] == ('table_test', 'role_test')

        # check the preamble was recorded
        metadata_name = get_metadata_name('table_test')
        metadata = get_table_data(cursor, metadata_name)
        assert metadata['table_preamble'][0] == 'table_preamble_value'

    def test_export_batch(self):
        tmp_folder = tempfile.mkdtemp()
        output_dir = os.path.join(tmp_folder, 'static')
        os.makedirs(os.path.join(output_dir, 'test_table'))

        connection = sqlite3.connect(':memory:')
        cursor = connection.cursor()

        batch = {
            'constant': 0,
            'value1': [1, 2, 3],
            'numpy_arrays': np.random.randn(3, 2000),
            'images': np.random.randint(0, 255, [3, 1, 128, 128]),
            'strings': ['p1', 'p2', 'p3'],
            'split': 'train',
            'dataset': 'test_data',
        }

        batch_copy = copy.deepcopy(batch)

        table_stream = TableStream(cursor, 'test_table', 'table_role_test')
        export_sample(tmp_folder, table_stream, 'basename', batch)

        # make sure the batch was NOT modified
        for name in batch_copy.keys():
            value_1 = batch[name]
            value_2 = batch_copy[name]
            if isinstance(value_1, np.ndarray):
                assert (value_1 == value_2).all()
            else:
                assert value_1 == value_2

        # check that images & large arrays are serialized
        files = os.listdir(os.path.join(output_dir, 'test_table'))

        assert 'basename_split_train_dataset_test_data_numpy_arrays_0.npy' in files
        assert 'basename_split_train_dataset_test_data_numpy_arrays_1.npy' in files
        assert 'basename_split_train_dataset_test_data_numpy_arrays_2.npy' in files

        # the correct values were serialized
        value_2 = np.load(os.path.join(output_dir, 'test_table', 'basename_split_train_dataset_test_data_numpy_arrays_2.npy'))
        assert (value_2 == batch['numpy_arrays'][2]).all()

        # check we have the correct type
        values = get_table_values(connection, 'test_table')
        names = table_stream.get_column_names()

        type_image = values[0][names.index(f'images{trw.reporting.SQLITE_TYPE_PATTERN}')]
        type_array = values[0][names.index(f'numpy_arrays{trw.reporting.SQLITE_TYPE_PATTERN}')]

        assert type_array == 'BLOB_NUMPY'
        assert type_image == 'BLOB_IMAGE_PNG'
        connection.commit()
        connection.close()

    def test_export_as_image_CXY(self):
        batch = {
            'images': list(np.random.randint(0, 255, [3, 1, 2, 2])),
        }

        tmp_folder = tempfile.mkdtemp()
        output_dir = os.path.join(tmp_folder, 'static')
        os.makedirs(output_dir)
        exported, attributes = trw.reporting.export_as_image(
            batch,
            'images',
            'image_0',
            0,
            tmp_folder, feature_attributes=None)
        assert exported
        assert attributes is not None
        min_value = attributes['min_value']
        max_value = attributes['max_value']

        files = os.listdir(output_dir)
        assert len(files) == 1
        assert 'image_0.png' in files

        # make sure the min/max values are the same for the normalization within batch
        exported, attributes = trw.reporting.export_as_image(
            batch,
            'images',
            'image_1',
            1,
            tmp_folder, feature_attributes=attributes)
        assert exported
        assert attributes is not None
        min_value2 = attributes['min_value']
        max_value2 = attributes['max_value']
        assert min_value2 == min_value
        assert max_value2 == max_value

        files = os.listdir(output_dir)
        assert len(files) == 2
        assert 'image_1.png' in files

    def test_table_with_missing(self):
        """
        Make sure we add data with columns not present in the table.

        If we add data without all columns, missing columns are populated with `None`
        """
        connection = sqlite3.connect(':memory:')
        cursor = connection.cursor()

        table = trw.reporting.TableStream(cursor, 'table_test', 'role_test', table_preamble='table_preamble_value')

        # create a single column
        batch = {
            'split': ['train'] * 3,
        }

        table.insert(batch)

        # insert a batch with new column. We expect the
        # existing rows with missing values will be set to None
        batch = {
            'split': ['test'] * 2,
            'epoch': [0, 1],
        }

        table.insert(batch)

        # insert a batch with missing column. We expect the
        # existing rows with missing values will be set to None
        batch = {
            'split': ['valid'] * 5,
        }

        table.insert(batch)

        data = get_table_data(cursor, 'table_test')
        assert len_batch(data) == 10
        assert len(data) == 2
        assert data['epoch'] == (None, None, None, 0, 1, None, None, None, None, None)
