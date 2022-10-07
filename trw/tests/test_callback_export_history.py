import unittest
import trw
import os
import glob


class TestCallbackExportHistory(unittest.TestCase):
    def test_basic(self):
        def make_step(dataset_name, split_name, task_name, measurements_dict):
            return {
                dataset_name: {
                    split_name: {
                        task_name: measurements_dict
                    }
                }
            }

        history = [
            make_step('dataset_1', 'split_1', 'task_1', {'measurement_1': 0.1, 'measurement_2': 0.2, 'measurement_3': 0.2}),
            make_step('dataset_1', 'split_1', 'task_1', {'measurement_1': 0.2, 'measurement_2': 0.15, 'measurement_3': 0.2, 'NOT_GOOD': 'String dont count!'}),
            make_step('dataset_1', 'split_1', 'task_1', {'measurement_1': 0.3, 'measurement_2': 0.5, 'measurement_3': 0.05, 'measurement_4': 1.0}),
        ]

        options = trw.train.Options()
        callback = trw.callbacks.CallbackExportHistory()
        outputs = {
            'dataset_1': {
                'split_1': {
                    'task_1': None
                }
            }
        }
        callback(options, history, None, None, outputs, None, None, None)

        expected_files = os.path.join(options.workflow_options.logging_directory, 'history', '*.png')
        matched_files = glob.glob(expected_files)
        assert len(matched_files) == 4


if __name__ == '__main__':
    unittest.main()
