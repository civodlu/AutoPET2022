from unittest import TestCase
import trw
import os

from trw.train.analysis_plots import gallery

import utils
import numpy as np


class TestPlots(TestCase):
    def test_history(self):
        # create 2 groups, with 1 group has 2 histories. Check this is displayed properly
        data = {
            'train': [[(0, 1.0), (1, 1.5), (2, 2.5),(3, 3.5),],
                      [(0, 1.1), (1, 1.6), (2, 2.6), (3, 3.6), ]],
            'valid': [[(0, 0.5), (1, 1.0), (3, 2.0), ]]
        }

        trw.train.plot_group_histories(utils.root_output, data, xlabel='epochs', ylabel='values', title='Values by epochs')
        assert os.path.exists(os.path.join(utils.root_output, 'Values by epochs.png'))

    def test_boxplots(self):
        data = {
            'feature_1': [0.8, 0.9, 0.85, 0.95],
            'feature_2': [0.7, 0.71, 0.72, 0.73]
        }

        trw.train.boxplots(utils.root_output, features_trials=data, xlabel='Features', ylabel='loss', title='Loss by features', meanline=True, y_range=[0.65, 1.0], rotate_x=45, showfliers=True)
        assert os.path.exists(os.path.join(utils.root_output, 'Loss by features.png'))

    def test_plot_roc(self):
        d = 100
        trues_1 = np.asarray([0] * d + [1] * d)
        trues_2 = np.asarray([0] * d + [1] * d)
        trues = [trues_1, trues_2]


        values_1 = np.random.rand(d * 2)
        values_2 = np.arange(0, d * 2, dtype=np.float) / (d * 2)
        values = [values_1, values_2]
        trw.train.plot_roc(utils.root_output, trues, values, 'ROC classifier 1 vs classifier 2', label_name=['Random classifier', 'Perfect Classifier'])

    def test_auroc_random(self):
        np.random.seed(0)
        d = 10000
        trues = np.random.randint(0, 1 + 1, d)
        random = (np.random.rand(d) - 0.5) * 10
        roc = trw.train.auroc(trues, random)
        assert abs(roc - 0.5) < 0.05  # random should be at 0.5

    def test_auroc_perfect(self):
        np.random.seed(0)
        d = 10000
        trues = np.random.randint(0, 1 + 1, d)
        roc = trw.train.auroc(trues, trues)
        assert abs(roc - 1.0) < 1e-4  # perfect classification -> AUC = 1.0

    def test_confusion_matrix(self):
        d = 100000
        trues = np.random.randint(0, 4 + 1, d)
        found = np.random.randint(0, 4 + 1, d)

        root = utils.root_output
        trw.train.confusion_matrix(
            root,
            found,
            trues,
            normalize=True,
            display_numbers=True,
            rotate_x=45,
            rotate_y=45,
            excludes_classes_with_samples_less_than=5,
            normalize_unit_percentage=True,
            title='test_confusion_matrix')
        assert os.path.exists(os.path.join(root, 'test_confusion_matrix.png'))

    def test_classification_report(self):
        d = 100
        trues = np.random.randint(0, 1 + 1, d)
        found = np.random.randn(d, 2)
        found_class = np.argmax(found, axis=1)
        class_mapping = {
            0: 'class_0',
            1: 'class_1',
        }
        r = trw.train.classification_report(found_class, found, trues, class_mapping)

        expected_accuracy = float(np.sum(found_class == trues)) / len(trues)
        assert abs(r['accuracy'] - expected_accuracy) < 1e-5

        fp = 0
        tp = 0
        tn = 0
        fn = 0
        for i in range(d):
            value_true = trues[i]
            value_found = found_class[i]
            if value_true == value_found:
                if value_true == 0:
                    tn += 1
                else:
                    tp += 1
            else:
                if value_true == 0:
                    fp += 1
                else:
                    fn += 1
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        assert abs(r['specificity'] - specificity) < 1e-5
        assert abs(r['sensitivity'] - sensitivity) < 1e-5

        cm_lines = [r['confusion_matrix'].replace('[', '').replace(']', '').split('\n')[n] for n in range(2)]
        cm_lines = [np.fromstring(line.strip(), sep=' ') for line in cm_lines]
        cm = np.asarray(cm_lines)
        assert cm[1, 1] == tp
        assert cm[0, 0] == tn
        assert cm[1, 0] == fn
        assert cm[0, 1] == fp

    def test_classification_report_binary(self):
        d = 100
        trues = np.random.randint(0, 1 + 1, d)
        found = np.random.rand(d)
        found_class = (found >= 0.5).astype(int)

        class_mapping = {
            0: 'class_0',
            1: 'class_1',
        }
        r = trw.train.classification_report(found_class, found, trues, class_mapping)

        expected_accuracy = float(np.sum(found_class == trues)) / len(trues)
        assert abs(r['accuracy'] - expected_accuracy) < 1e-5

        fp = 0
        tp = 0
        tn = 0
        fn = 0
        for i in range(d):
            value_true = trues[i]
            value_found = found_class[i]
            if value_true == value_found:
                if value_true == 0:
                    tn += 1
                else:
                    tp += 1
            else:
                if value_true == 0:
                    fp += 1
                else:
                    fn += 1
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        assert abs(r['specificity'] - specificity) < 1e-5
        assert abs(r['sensitivity'] - sensitivity) < 1e-5

        cm_lines = [r['confusion_matrix'].replace('[', '').replace(']', '').split('\n')[n] for n in range(2)]
        cm_lines = [np.fromstring(line.strip(), sep=' ') for line in cm_lines]
        cm = np.asarray(cm_lines)
        assert cm[1, 1] == tp
        assert cm[0, 0] == tn
        assert cm[1, 0] == fn
        assert cm[0, 1] == fp

    def test_gallery(self):
        images = [
            # variably sized images
            [np.zeros([64, 64, 3]), np.zeros([64, 64, 3]), np.zeros([64, 64, 3])],
            [np.zeros([32, 64, 3]), np.zeros([32, 64, 3]), np.zeros([32, 64, 3])],
        ]
        x_axis_text = ['X0', 'X1', 'X2']
        y_axis_text = ['Y0', 'Y1']
        title = 'Gallery of empty images'
        root = utils.root_output
        save_path = os.path.join(root, 'empty_gallery.png')
        gallery(images, x_axis_text, y_axis_text, title, save_path)
        assert os.path.exists(save_path)
