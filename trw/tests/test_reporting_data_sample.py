import os
import sqlite3
import tempfile
from unittest import TestCase
import trw.reporting
import numpy as np
from bokeh.models import Div, DataTable, Panel, ImageURL, Rect
from bokeh.plotting import Figure

from trw.reporting import TableStream, export_sample
from trw.reporting.bokeh_ui import BokehUi
from trw.reporting.normalize_data import normalize_data
from trw.reporting.reporting_bokeh_samples import process_data_samples
from trw.reporting.data_category import DataCategory
from trw.reporting.reporting_bokeh_tabs_dynamic_data import TabsDynamicData, get_data_normalize_and_alias, \
    create_aliased_table
from trw.reporting.reporting_bokeh_tabs_dynamic_header import TabsDynamicHeader
from trw.reporting.table_sqlite import get_table_data, get_tables_name_and_role


def make_table(cursor, table_name, table_role, batch, db_path=None):
    if db_path is None:
        tmp_folder = tempfile.mkdtemp()
    else:
        tmp_folder = os.path.dirname(db_path)
    output_dir = os.path.join(tmp_folder, 'static')
    os.makedirs(os.path.join(output_dir, table_name))

    table_stream = TableStream(cursor, table_name, table_role)
    export_sample(tmp_folder, table_stream, 'basename', batch)
    return tmp_folder, table_stream


def find_named(ui, name):
    """
    Go through the hierarchy of UI elements and find the elements that have the same name
    """
    results = set()
    if hasattr(ui, 'name'):
        if ui.name == name:
            results.add(ui)

    if hasattr(ui, 'children'):
        for c in ui.children:
            resuls_children = find_named(c, name)
            results.update(resuls_children)

    if hasattr(ui, 'tabs'):
        for c in ui.tabs:
            resuls_children = find_named(c, name)
            results.update(resuls_children)

    if hasattr(ui, 'child'):
        resuls_children = find_named(ui.child, name)
        results.update(resuls_children)

    return results


class PeriodicCallbacks:
    def __init__(self):
        self.callback = None

    def add_periodic_callback(self, callback, refresh_time):
        self.callback = callback

    def execute(self):
        self.callback()


class TabDataUpdate(BokehUi):
    def __init__(self):
        self.updated = False
        super().__init__(ui=Panel(name='TEST'))

    def update_data(self, options, name, data, types, type_categories):
        self.updated = True


class TestReporting(TestCase):
    def test_data_normalization_column_expansion(self):
        # multi dimensional numpy arrays can be expanded as a single columns
        # to facilitate analysis
        connection = sqlite3.connect(':memory:')
        cursor = connection.cursor()

        batch = {
            'numpy_arrays_int_5': np.random.randint(0, 10, [3, 5]),
        }

        table_name = 'table_name'
        tmp_folder , _ = make_table(cursor, table_name, 'table_role', batch)

        data = get_table_data(cursor, table_name)
        options = trw.reporting.create_default_reporting_options()
        options.db_root = os.path.join(tmp_folder, 'test.db')

        normalized_data, types, type_categories = normalize_data(options, data, table_name)
        assert len(normalized_data) == 5
        assert 'numpy_arrays_int_5_0' in normalized_data
        assert 'numpy_arrays_int_5_1' in normalized_data
        assert 'numpy_arrays_int_5_2' in normalized_data
        assert 'numpy_arrays_int_5_3' in normalized_data
        assert 'numpy_arrays_int_5_4' in normalized_data
        assert type_categories['numpy_arrays_int_5_4'] == DataCategory.DiscreteOrdered

    def test_data_normalization_different_numpy_shapes_3d(self):
        # make sure we can load differently shaped data samples
        connection = sqlite3.connect(':memory:')
        cursor = connection.cursor()

        batch = {
            'values': [
                np.zeros([1, 1, 64, 64, 64]),
                np.zeros([1, 1, 64, 64, 64]),
                np.zeros([1, 1, 65, 65, 65])],
        }

        table_name = 'table_name'
        tmp_folder, _ = make_table(cursor, table_name, 'table_role', batch)

        data = get_table_data(cursor, table_name)
        options = trw.reporting.create_default_reporting_options()
        options.db_root = os.path.join(tmp_folder, 'test.db')

        normalized_data, types, type_categories = normalize_data(options, data, table_name)
        assert len(normalized_data['values']) == 3
        assert normalized_data['values'][0].shape == (1, 1, 64, 64, 64)
        assert normalized_data['values'][2].shape == (1, 1, 65, 65, 65)

    def test_data_normalization_different_numpy_shapes_2d(self):
        # make sure we can load differently shaped data samples
        connection = sqlite3.connect(':memory:')
        cursor = connection.cursor()

        batch = {
            'values': [
                np.zeros([1, 1, 64, 64]),
                np.zeros([1, 1, 64, 64]),
                np.zeros([1, 1, 65, 65])],
        }

        table_name = 'table_name'
        tmp_folder, _ = make_table(cursor, table_name, 'table_role', batch)

        data = get_table_data(cursor, table_name)
        options = trw.reporting.create_default_reporting_options()
        options.db_root = os.path.join(tmp_folder, 'test.db')

        normalized_data, types, type_categories = normalize_data(options, data, table_name)
        assert len(normalized_data['values']) == 3

        # these were exported as image
        assert '0.png' in normalized_data['values'][0]
        assert '1.png' in normalized_data['values'][1]
        assert '2.png' in normalized_data['values'][2]

    def test_data_normalization(self):
        # publish data the same way the application would:
        # export samples to a database, then read the data
        connection = sqlite3.connect(':memory:')
        cursor = connection.cursor()

        batch = {
            'constant': 0,
            'value_continuous': [1.1, 2.2, 3.3],
            'value_integers': [1, 2, 3],
            'numpy_arrays': np.random.randn(3, 2000),
            'numpy_arrays_int_10': np.random.randint(0, 10, [3, 10]),
            'numpy_arrays_int_1': np.random.randint(0, 10, [3]),
            'images': np.random.randint(0, 255, [3, 3, 128, 128]),
            'strings': ['p1', 'p2', 'p3'],
            'split': 'train',
            'column_to_be_removed': ['p1', 'p2', 'p3'],
        }

        table_name = 'table_name'
        tmp_folder , _ = make_table(cursor, table_name, 'table_role', batch)

        subsampling_factor = 2 / 3
        data = get_table_data(cursor, table_name)
        options = trw.reporting.create_default_reporting_options()
        options.config = {
            'table_name': {
                'data': {
                    'subsampling_factor': subsampling_factor,
                    'remove_columns': ['column_to_be_removed']
                }
            }
        }
        options.data.unpack_numpy_arrays_with_less_than_x_columns = 3
        options.db_root = os.path.join(tmp_folder, 'test.db')

        # must have sub-sampled the data
        normalized_data, types, type_categories = normalize_data(options, data, table_name)
        assert len(normalized_data['constant']) == 2, 'subsampling failed!'

        # must have exactly the batch keys
        assert len(batch) == len(normalized_data) + 1
        for key in normalized_data.keys():
            assert key in batch
        assert 'column_to_be_removed' not in normalized_data.keys()

        # for BLOB_IMAGE, must have appended the `appname` (folder of the datbase name)
        appname = os.path.basename(os.path.dirname(options.db_root))
        assert appname in normalized_data['images'][0]

        # images must be served from a static directory
        assert 'static' in normalized_data['images'][0]

        assert type_categories['images'] == DataCategory.Other
        assert type_categories['numpy_arrays'] == DataCategory.Other  # too many dimensions
        assert type_categories['numpy_arrays_int_10'] == DataCategory.Other
        assert type_categories['numpy_arrays_int_1'] == DataCategory.DiscreteOrdered
        assert type_categories['value_continuous'] == DataCategory.Continuous
        assert type_categories['value_integers'] == DataCategory.DiscreteOrdered
        assert type_categories['strings'] == DataCategory.DiscreteUnordered
        assert type_categories['split'] == DataCategory.DiscreteUnordered

        assert types['numpy_arrays'] == 'BLOB_NUMPY'
        assert types['images'] == 'BLOB_IMAGE_PNG'
        assert types['numpy_arrays_int_10'] == 'BLOB_NUMPY'

    def test_table_data_samples(self):
        # make sure we can render the samples tabs with tabular and scatter data
        connection = sqlite3.connect(':memory:')
        cursor = connection.cursor()

        batch = {
            'value_continuous': [1.1, 2.2, 3.3],
            'value_integers': [1, 2, 3],
            'numpy_arrays': np.random.randn(3, 2000),
            'numpy_arrays_int_1': np.random.randint(0, 10, [3]),
            'images': np.random.randint(0, 255, [3, 3, 128, 128]),
            'strings': ['p1', 'p2', 'p3'],
            'column_to_be_removed': ['p1', 'p2', 'p3'],
        }

        table_name = 'table_name'
        tmp_folder , _ = make_table(cursor, table_name, 'table_role', batch)

        options = trw.reporting.create_default_reporting_options()
        options.db_root = os.path.join(tmp_folder, 'test.db')

        d, types, type_categories = normalize_data(options, batch, table_name)
        tabs = process_data_samples(options, 'table_name', 'table_role', d, types, type_categories)

        assert len(tabs) == 2  # must have the `tabular` and `scatter` tabs

        #
        # Check the tabular tab
        #
        assert len(tabs[0].ui.child.children) == 2  # must have a DataTable, Div
        table = tabs[0].ui.child.children[0]
        assert isinstance(table, DataTable)
        div = tabs[0].ui.child.children[1]
        assert isinstance(tabs[0].ui.child.children[1], Div)    # DIV to configure extra CSS
                                                                # for the table (text rotation)
        assert div.visible is False

        assert len(table.source.column_names) == len(batch)

        #
        # Check the scatter tab
        #
        head = tabs[1].ui.child.children
        assert len(head[0].children) >= 7  # all the tools

        figures = head[1].children
        assert len(figures) == 1  # should have a single figure
        assert isinstance(figures[0].children[0], Figure)
        assert len(figures[0].children[0].renderers) > 0  # if not, failure!

    def test_table_data_samples__default_config__discrete_discrete(self):
        # make sure we can configure defaults
        connection = sqlite3.connect(':memory:')
        cursor = connection.cursor()

        batch = {
            'value_continuous': [1.1, 2.2, 3.3],
            'value_integers': [1, 2, 3],
            'numpy_arrays': np.random.randn(3, 2000),
            'numpy_arrays_int_1': np.random.randint(0, 10, [3]),
            'images': np.random.randint(0, 255, [3, 3, 128, 128]),
            'strings': ['p1', 'p2', 'p3'],
        }

        table_name = 'table_name'
        tmp_folder , _ = make_table(cursor, table_name, 'table_role', batch)

        config = {
            'table_name': {
                'default': {
                    'Scatter X Axis': 'value_integers',
                    'Scatter Y Axis': 'strings',
                    'Binning X Axis': 'value_integers',
                    'Color by': 'value_continuous',
                    'Display with': 'Dot'
                }
            }
        }

        options = trw.reporting.create_default_reporting_options(config=config)
        options.db_root = os.path.join(tmp_folder, 'test.db')

        d, types, type_categories = normalize_data(options, batch, table_name)
        tabs = process_data_samples(options, 'table_name', 'table_role', d, types, type_categories)

        #
        # Check the scatter tab
        #
        head = tabs[1].ui.child.children
        assert len(head) >= 2  # Tools, Scatter 1+2+3

        assert head[0].children[0].value == 'value_integers'
        assert head[0].children[1].value == 'strings'
        assert head[0].children[2].value == 'value_continuous'
        print('DONE')

    def test_table_data_samples__scatter_and_discrete_configs(self):
        # make sure we can configure defaults
        connection = sqlite3.connect(':memory:')
        cursor = connection.cursor()

        batch = {
            'value_continuous': [1.1, 2.2, 3.3],
            'value_integers': [1, 2, 3],
            'numpy_arrays': np.random.randn(3, 2000),
            'numpy_arrays_int_1': np.random.randint(0, 10, [3]),
            'images': np.random.randint(0, 255, [3, 3, 128, 128]),
            'strings': ['p1', 'p2', 'p3'],
        }

        table_name = 'table_name'
        tmp_folder, _ = make_table(cursor, table_name, 'table_role', batch)

        #
        # X: continuous, Y: discrete
        #
        config = {
            'table_name': {
                'default': {
                    'Scatter X Axis': 'value_continuous',
                    'Scatter Y Axis': 'value_integers',
                    'Color by': 'strings',
                    'Display with': 'Dot'
                }
            }
        }

        options = trw.reporting.create_default_reporting_options(config=config)
        options.db_root = os.path.join(tmp_folder, 'test.db')

        d, types, type_categories = normalize_data(options, batch, table_name)
        tabs = process_data_samples(options, 'table_name', 'table_role', d, types, type_categories)
        head = tabs[1].ui.child.children
        assert len(head) == 2  # Tools, row(Fig, Figure-color-legend)
        assert len(head[1].children) == 2  # Fig, Figure-color-legend
        assert len(head[1].children[0].children[0].renderers) > 0

        #
        # Y: continuous, X: discrete
        #
        config = {
            'table_name': {
                'default': {
                    'Scatter X Axis': 'value_integers',
                    'Scatter Y Axis': 'value_continuous',
                    'Color by': 'strings',
                    'Display with': 'Dot'
                }
            }
        }

        options = trw.reporting.create_default_reporting_options(config=config)
        options.db_root = os.path.join(tmp_folder, 'test.db')

        tabs = process_data_samples(options, 'table_name', 'table_role', d, types, type_categories)
        head = tabs[1].ui.child.children
        assert len(head) == 2  # Tools, row(Fig, Figure-color-legend)
        assert len(head[1].children[0].children[0].renderers) > 0

        #
        # Y: continuous, X: continuous
        #
        config = {
            'table_name': {
                'default': {
                    'Scatter X Axis': 'value_continuous',
                    'Scatter Y Axis': 'value_continuous',
                    'Color by': 'strings',
                    'Display with': 'Dot'
                }
            }
        }

        options = trw.reporting.create_default_reporting_options(config=config)
        options.db_root = os.path.join(tmp_folder, 'test.db')

        tabs = process_data_samples(options, 'table_name', 'table_role', d, types, type_categories)
        head = tabs[1].ui.child.children
        assert len(head) == 2  # Tools, Fig, Figure-color-legend
        assert len(head[1].children[0].children[0].renderers) > 0

    def test_header_update(self):
        """
        Test ``TabsDynamicHeader`` responds to adding new tables in the SQL database
        """
        connection = sqlite3.connect(':memory:')
        cursor = connection.cursor()

        options = trw.reporting.create_default_reporting_options()
        options.db_root = 'NONE'

        doc = PeriodicCallbacks()
        header_updated = False

        def creator_fn(name, role):
            nonlocal header_updated
            header_updated = True
            return Panel()

        dynamic_header = TabsDynamicHeader(doc, options, connection, creator_fn)
        doc.execute()

        # there is not change, should not be updated
        assert not header_updated

        # create a table, this should trigger the callback
        table_name = 'table_name'
        batch = {'value_continuous': [1.1, 2.2, 3.3]}
        make_table(cursor, table_name, 'table_role', batch)
        doc.execute()
        assert header_updated

    def test_data_update(self):
        """
        Test ``TabsDynamicData`` responds to update of an underlying SQL table
        """
        connection = sqlite3.connect(':memory:')
        cursor = connection.cursor()

        options = trw.reporting.create_default_reporting_options()
        options.db_root = 'NONE'

        doc = PeriodicCallbacks()
        tabs = None

        def creator_fn(options, name, role, data, types, type_categories):
            # create the dynamic tabs
            nonlocal tabs
            tabs = [TabDataUpdate()]
            return tabs

        table_name = 'table_name'
        table_role = 'table_role'

        # create a table, this should not trigger the callback
        batch = {'value_continuous': [1.1, 2.2, 3.3]}
        table, table_stream = make_table(cursor, table_name, table_role, batch)
        dynamic_data = TabsDynamicData(doc, options, connection, table_name, table_role, creator_fn)
        doc.execute()
        assert len(tabs) == 1
        assert not tabs[0].updated

        # no data update, should not trigger the callback
        doc.execute()
        assert len(tabs) == 1
        assert not tabs[0].updated

        # update the table, this should trigger the callback
        table_stream.insert(batch)
        number_of_rows = trw.reporting.get_table_number_of_rows(cursor, table_name)
        assert number_of_rows == 6

        # table was modified, expect the callback to be triggered
        doc.execute()
        assert len(tabs) == 1
        assert tabs[0].updated

    def test_table_aliasing(self):
        """
        Access data of an aliased table. Aliased table allow us to create
        table with different configurations (e.g., different view of the data) and
        not repeating the tables.
        """

        # set up the real table
        connection = sqlite3.connect(':memory:')
        cursor = connection.cursor()
        batch = {'value_continuous': [1.1, 2.2, 3.3]}
        table, table_stream = make_table(cursor, 'real_table', 'table_role', batch)

        # create an aliased table
        aliased_name = 'aliased_table'
        create_aliased_table(connection, aliased_name, 'real_table')

        # retrieve the data
        options = trw.reporting.create_default_reporting_options()
        options.db_root = 'NONE'
        data, types, type_categories, alias = get_data_normalize_and_alias(options, connection, aliased_name)
        assert 'value_continuous' in data
        assert alias == 'real_table'

    def test_scatter_default_view_with_image(self):
        # set up the real table
        tmp_folder = tempfile.mkdtemp()
        db_path = os.path.join(tmp_folder, 'database.db')

        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        batch = {
            'value_continuous': np.arange(10),
            'images': np.ones((10, 1, 32, 32), dtype=np.float32)
        }
        tmp_folder, _ = make_table(cursor, 'test_table', 'data_samples', batch, db_path=db_path)
        connection.commit()

        options = trw.reporting.create_default_reporting_options()
        options.image_size = 64
        doc = trw.reporting.report(db_path, options)
        assert len(doc.roots) == 1

        controls = find_named(doc.roots[0], 'PanelDataSamplesScatter_controls')
        uis = find_named(doc.roots[0], 'data_samples_fig')
        icon = find_named(doc.roots[0], 'icon')

        assert len(controls) == 1, 'expected a single control with the scatter plot'
        assert next(iter(icon)).value == 'images', 'we have an image like array. This should be used as default'
        assert len(uis) == 1, 'expected a single figure'

        # make sure we are displaying images & caption
        fig = next(iter(uis))
        render_images = fig.renderers[0]
        assert isinstance(render_images.glyph, ImageURL), 'expected Image URL'
        render_caption = fig.renderers[1]
        assert isinstance(render_caption.glyph, Rect), 'expected Rect (image caption)'

        # check the image coordinate: no filter, ordered by index
        xs = render_images.data_source.data['fig_0_data_x']
        ys = render_images.data_source.data['fig_0_data_y']

        assert (xs == [0, 64, 128, 192, 0, 64, 128, 192, 0, 64]).all()
        assert (ys == [0, 0, 0, 0, 64, 64, 64, 64, 128, 128]).all()
