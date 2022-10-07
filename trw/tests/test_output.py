from unittest import TestCase

import trw
import trw.train
import numpy as np
import torch
import functools
import torch.nn as nn
import trw.utils
from trw.train import segmentation_criteria_ce_dice, LossDiceMulticlass
from trw.train.metrics import MetricSegmentationDice, MetricLoss, MetricClassificationError, \
    MetricClassificationBinarySensitivitySpecificity, MetricClassificationBinaryAUC, MetricClassificationF1
from trw.train.trainer import generic_aggregate_loss_terms


class TestOutput(TestCase):
    def test_regression_0_loss(self):
        input_values = torch.from_numpy(np.asarray([[1, 2], [3, 4]], dtype=float))
        target_values = torch.from_numpy(np.asarray([[1, 2], [3, 4]], dtype=float))

        o = trw.train.OutputRegression(input_values, target_values)
        batch = {'target': target_values}
        r = o.evaluate_batch(batch, False)

        loss = r['loss'].data.numpy()
        self.assertTrue(len(loss.shape) == 0)
        self.assertTrue(loss == 0.0)

    def test_regression_error(self):
        input_values = torch.from_numpy(np.asarray([[1, 2], [3, 4]], dtype=float))
        target_values = torch.from_numpy(np.asarray([[0, 0], [0, 0]], dtype=float))

        o = trw.train.OutputRegression(input_values, target_values)
        batch = {'target': target_values}
        r = o.evaluate_batch(batch, False)

        loss = r['loss'].data.numpy()
        self.assertTrue(len(loss.shape) == 0)
        self.assertTrue(loss == (1+2*2+3*3+4*4) / 4)

    def test_regression_scaling(self):
        input_values = torch.from_numpy(np.asarray([[1, 2], [3, 4]], dtype=float))
        target_values = torch.from_numpy(np.asarray([[0, 0], [0, 0]], dtype=float))

        o = trw.train.OutputRegression(input_values, target_values, loss_scaling=2.0)
        batch = {'target': target_values}
        r = o.evaluate_batch(batch, False)

        loss = r['loss'].data.numpy()
        self.assertTrue(len(loss.shape) == 0)
        self.assertTrue(loss == (1+2*2+3*3+4*4) / 4 * 2)

    def test_regression_weight(self):
        input_values = torch.from_numpy(np.asarray([[1, 2], [3, 4]], dtype=float))
        target_values = torch.from_numpy(np.asarray([[0, 0], [0, 0]], dtype=float))
        weights = torch.from_numpy(np.asarray([1, 0], dtype=float))

        o = trw.train.OutputRegression(input_values, target_values, weights=weights)
        batch = {'target': target_values, 'weights': weights}
        r = o.evaluate_batch(batch, False)

        loss = r['loss'].data.numpy()
        self.assertTrue(len(loss.shape) == 0)
        self.assertTrue(loss == (1+2*2+3*3*0+4*4*0) / 4 * 2)

    def test_classification_0_loss(self):
        input_values = torch.from_numpy(np.asarray([[0.0, 100.0], [100.0, 0.0]], dtype=float))
        target_values = torch.from_numpy(np.asarray([1, 0], dtype=np.int64))

        o = trw.train.OutputClassification(input_values, target_values, classes_name='target')
        batch = {'target': target_values}
        r = o.evaluate_batch(batch, False)

        loss = r['loss'].data.numpy()
        self.assertTrue(len(loss.shape) == 0)
        self.assertTrue(loss < 1e-6)

        assert r.get('output') is not None
        self.assertTrue((trw.utils.to_value(r['output']).squeeze(1) == [1, 0]).all())

    def test_classification_weight(self):
        input_values = torch.from_numpy(np.asarray([[0.0, 100.0], [100.0, 0.0]], dtype=float))
        target_values = torch.from_numpy(np.asarray([1, 1], dtype=np.int64))
        weights = torch.from_numpy(np.asarray([1, 0], dtype=float))

        o = trw.train.OutputClassification(input_values, target_values, classes_name='target', weights=weights)
        batch = {'target': target_values, 'weights': weights}
        r = o.evaluate_batch(batch, False)

        loss = r['loss'].data.numpy()
        self.assertTrue(len(loss.shape) == 0)
        self.assertTrue(loss < 1e-6)

    def test_output_segmentation(self):
        mask_scores = torch.zeros(10, 4, 32, 32, dtype=torch.float32)  # 10 samples, 4 classes, 2D 32x32 grid
        mask_scores[:, 1, :, :] = 100.0
        expected_map = torch.ones(10, 1, 32, 32, dtype=torch.int64)

        o = trw.train.OutputSegmentation(
            output=mask_scores,
            output_truth=expected_map,
            weights=torch.ones(10, dtype=torch.float32),
            criterion_fn=lambda: functools.partial(trw.train.segmentation_criteria_ce_dice, ce_weight=0.9999999))
        batch = {
            'target': expected_map,
        }
        loss_term = o.evaluate_batch(batch, is_training=False)
        assert trw.utils.to_value(loss_term['loss']) < 1e-5
        assert (trw.utils.to_value(loss_term['output']) == 1).all()

    def test_output_segmentation_weight(self):
        """
        Make sure the loss can be weighted with a per-voxel mask
        """
        torch.manual_seed(0)
        mask_scores = torch.randn([10, 4, 32, 40], dtype=torch.float32)  # 10 samples, 4 classes, 2D 32x32 grid
        expected_map = torch.ones(10, 1, 32, 40, dtype=torch.int64)

        mask_weight = torch.zeros_like(expected_map, dtype=torch.float32)
        mask_weight[:, :, 4:21, 8:38] = 1

        def loss(found, expected, per_voxel_weights):
            l = mask_weight.view(10, 1, 32, 40) * found
            return l.sum([1, 2, 3])

        o = trw.train.OutputSegmentation(
            output=mask_scores,
            output_truth=expected_map,
            per_voxel_weights=mask_weight,
            criterion_fn=lambda: loss)

        batch = {
            'target': expected_map,
        }

        loss_term = o.evaluate_batch(batch, is_training=False)
        expected_loss = mask_scores[:, :, 4:21, 8:38].sum([1, 2, 3])

        losses = loss_term['losses']
        assert torch.max(torch.abs(losses - expected_loss)) < 1e-3

        # make sure we support per_voxel_weights arguments
        o = trw.train.OutputSegmentation(
            output=mask_scores,
            output_truth=expected_map,
            per_voxel_weights=torch.zeros_like(expected_map, dtype=torch.float32),
            criterion_fn=lambda: functools.partial(segmentation_criteria_ce_dice, ce_weight=1.0))
        loss_term = o.evaluate_batch(batch, is_training=False)
        assert loss_term['loss'].item() == 0.0

    def test_output_triplets(self):
        samples = torch.tensor(
            [
                [[[1, 2]]],  # make sure it works for N-d features
                [[[1, 3]]],
                [[[1, 4]]],
            ], dtype=torch.float32)

        samples_p = torch.tensor(
            [
                [[[1, 2.1]]],
                [[[1, 3.1]]],
                [[[1, 4.1]]],
            ], dtype=torch.float32)

        samples_n = torch.tensor(
            [
                [[[1, 20.1]]],  # far from the samples_p and margin, should be 0 loss
                [[[1, 3.1]]],  # not satisfying the margin
                [[[1, 40.1]]],  # far from the samples_p and margin, should be 0 loss
            ], dtype=torch.float32)

        weights = torch.tensor([1.0, 3.0, 1.0])
        o = trw.train.OutputTriplets(samples, samples_p, samples_n, weight_name='weights')
        loss_term = o.evaluate_batch({'weights': weights}, is_training=False)

        assert loss_term['loss'] == 1.0
        assert loss_term['output_raw'] is samples
        assert isinstance(loss_term['output_ref'], trw.train.OutputTriplets)

    def test_output_loss(self):
        losses = torch.tensor(
            [
                [[[1, 2]]],  # make sure it works for N-d features
                [[[1, 3]]],
                [[[1, 4]]],
            ], dtype=torch.float32)

        batch = {
            'uid': ['s0', 's1', 's2']
        }

        output = trw.train.OutputLoss(losses, sample_uid_name='uid')
        loss_term = output.evaluate_batch(batch, False)

        assert loss_term['loss'] == torch.mean(losses)
        assert loss_term['losses'] is losses
        assert loss_term['output_ref'] is output
        assert loss_term['uid'] is batch['uid']

    def test_output_classification_no_criterion(self):
        input_values = torch.from_numpy(np.asarray([[0.0, 100.0], [100.0, 0.0]], dtype=float))
        target_values = torch.from_numpy(np.asarray([1, 0], dtype=np.int64))

        o = trw.train.OutputClassification(input_values, target_values, classes_name='target', criterion_fn=None)
        batch = {'target': target_values}
        r = o.evaluate_batch(batch, False)

        loss = r['loss'].data.numpy()
        self.assertTrue(loss == 0)

    def test_classification_multi_dim(self):
        i1 = np.asarray([
            [
                [1, 1, 0],
                [0, 0, 0],
            ], [
                [0, 0, 1],
                [1, 1, 1],
            ]
        ], dtype=np.float32)

        o1 = np.asarray([
                [0, 0, 1],
                [1, 1, 1],
            ], dtype=np.int64)

        i2 = np.asarray([
            [
                [0, 0, 0],
                [1, 0, 0],
            ], [
                [1, 1, 1],
                [0, 1, 1],
            ]
        ], dtype=np.float32)

        o2 = np.asarray([
            [1, 1, 1],
            [0, 1, 1],
        ], dtype=np.int64)

        inputs = torch.from_numpy(np.asarray([i1, i2]))
        outputs = torch.from_numpy(np.asarray([o1, o2]))
        batch = {
            'classes': outputs,
        }
        o = trw.train.OutputClassification(inputs, batch['classes'], classes_name='classes')
        os = o.evaluate_batch(batch, is_training=True)

        nb_trues = int((os['output_truth'] == os['output']).sum())
        assert nb_trues == 2 * 2 * 3

        h_classification = None
        for metric, h in os['metrics_results'].items():
            if isinstance(metric, trw.train.MetricClassificationError):
                h_classification = h
        assert h_classification['nb_trues'] == 12

    def test_binary_segmentation(self):
        i1 = [
            [1000, 0.01, 0.0001],
            [-0.8, -0.5, -10000],
        ]

        o1 = [
            [0, 1, 1],
            [1, 0, 0],
        ]

        i2 = [
            [-0.0001, 1000, 0.01],
            [-0.1, -0.2, -0.3],
        ]

        o2 = [
            [0, 1, 1],
            [0, 0, 0],
        ]

        # NCX format
        i1 = torch.as_tensor(i1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        i2 = torch.as_tensor(i2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        o1 = torch.as_tensor(o1, dtype=torch.long).unsqueeze(0).unsqueeze(0)
        o2 = torch.as_tensor(o2, dtype=torch.long).unsqueeze(0).unsqueeze(0)

        metrics = [MetricSegmentationDice()]
        criterion_fn = functools.partial(LossDiceMulticlass, smooth=0, power=1, eps=0, discard_background_loss=False, normalization_fn=nn.Sigmoid)

        output = trw.train.OutputSegmentationBinary(i1, o1, metrics=metrics, criterion_fn=criterion_fn)
        v1 = output.evaluate_batch({}, is_training=True)

        assert (v1['output_raw'] == i1).all()
        expected_o1 = torch.tensor([
            [1, 1, 1],
            [0, 0, 0]
        ], dtype=torch.long)
        assert (v1['output'] == expected_o1).all()

        # calculate the dice (i1)
        p = torch.sigmoid(i1)
        intersection = (p * o1.float()).sum()
        cardinality = p.sum() + o1.sum()
        loss = 1 - 2 * intersection / cardinality
        assert abs(loss - v1['loss']) < 1e-4

        metric = list(v1['metrics_results'].values())[0]
        assert abs(metric['numerator'].squeeze() - 4) < 1e-4
        assert abs(metric['cardinality'].squeeze() - 6) < 1e-4

        output = trw.train.OutputSegmentationBinary(i2, o2, metrics=metrics, criterion_fn=criterion_fn)
        v2 = output.evaluate_batch({}, is_training=True)
        expected_o2 = torch.tensor([
            [0, 1, 1],
            [0, 0, 0]
        ], dtype=torch.long)
        assert (v2['output'] == expected_o2).all()

        p = torch.sigmoid(i2)
        intersection = (p * o2.float()).sum()
        cardinality = p.sum() + o2.sum()
        loss = 1 - 2 * intersection / cardinality
        assert abs(loss - v2['loss']) < 1e-4

        metric = list(v2['metrics_results'].values())[0]
        assert abs(metric['numerator'].squeeze() - 4) < 1e-4
        assert abs(metric['cardinality'].squeeze() - 4) < 1e-4

        output_metric = metrics[0].aggregate_metrics([
            list(v1['metrics_results'].values())[0],
            list(v2['metrics_results'].values())[0]
        ])

        output_metric_expected = 1 - (4 + 4) / (6 + 4)
        assert abs(output_metric['1-dice'] - output_metric_expected) < 1e-4
        assert abs(output_metric['1-dice[class=0]'] - output_metric_expected) < 1e-4

    def test_binary_classification(self):
        i1 = [1000, -0.01, -0.0001, 0.01, 0.51, -10000]
        i1 = torch.tensor(i1).unsqueeze(1)

        o1 = [1, 0, 0, 1, 1, 0]
        o1 = torch.tensor(o1).unsqueeze(1)

        metrics_fns = [
            MetricLoss(),
            MetricClassificationError(),
            MetricClassificationBinarySensitivitySpecificity(),
            MetricClassificationBinaryAUC(),
            MetricClassificationF1(),
        ]
        output = trw.train.OutputClassificationBinary(i1, o1, metrics=metrics_fns)
        v2 = output.evaluate_batch({}, is_training=True)
        assert (v2['output'] == o1).all()

        _, aggregated_metrics = generic_aggregate_loss_terms([{'test': v2}])
        assert aggregated_metrics['test']['classification error'] == 0
        assert aggregated_metrics['test']['1-f1[binary]'] == 0
        assert aggregated_metrics['test']['1-sensitivity'] == 0
        assert aggregated_metrics['test']['1-specificity'] == 0
        assert aggregated_metrics['test']['1-auc'] == 0
