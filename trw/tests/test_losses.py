import sklearn
import trw.utils

import trw
import numpy as np
import torch
import torch.nn as nn
from unittest import TestCase
import scipy

from trw.train import LossContrastive
from trw.train.losses import LossCrossEntropyCsiMulticlass
from trw.train.metrics import MetricSegmentationDice


class IdNetCenterLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.center_loss = trw.train.LossCenter(2, 2)

    def forward(self, batch):
        features = batch['features']
        classes = batch['classes']
        center_loss = self.center_loss(features, classes)
        return trw.train.OutputLoss(center_loss)


class TestLosses(TestCase):
    def test_losses_dice_single_binary_sample(self):
        found_channel0 = [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]

        found_channel1 = [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]

        targets = [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
        ]

        t1 = torch.from_numpy(np.asarray([found_channel0, found_channel1], dtype=np.float32).reshape((1, 2, 5, 6)))
        t2 = torch.from_numpy(np.asarray(targets, dtype=np.int64).reshape((1, 1, 5, 6)))

        loss = trw.train.LossDiceMulticlass(normalization_fn=None, eps=1e-7, smooth=0, discard_background_loss=False)
        l = loss(t1, t2)

        # calculate the mask intersection
        # we must have a mask for background & foreground
        num_intersection_0 = 4
        num_intersection_1 = 6

        # calculate the number of pixels ON of segmentation of background & foreground
        num_output_0 = 18
        num_output_1 = 12

        num_target_0 = 10
        num_target_1 = 20

        # calculate the expected dice score
        expected_dice_0 = 2 * num_intersection_0 / (num_output_0 + num_target_0)
        expected_dice_1 = 2 * num_intersection_1 / (num_output_1 + num_target_1)

        # transform this to a loss that we want to minimize
        expected_loss = 1.0 - (expected_dice_0 + expected_dice_1) / 2.0

        assert abs(expected_loss - float(l)) < 1e-4

    def test_losses_dice_multiple_binary_sample(self):
        targets = [
            [[1, 1, 0],
             [1, 1, 0]],

            [[0, 1, 1],
             [0, 1, 1]],
        ]

        sample_0 = [
            # channel 0
            [[1, 1, 1],
             [0, 0, 0]],

            # channel 1
            [[0, 0, 0],
             [1, 1, 1]],
        ]

        sample_1 = [
            # channel 0
            [[1, 1, 0],
             [1, 1, 0]],

            # channel 1
            [[0, 0, 1],
             [0, 0, 1]],
        ]

        t1 = torch.from_numpy(np.asarray([sample_0, sample_1], dtype=np.float32).reshape((2, 2, 2, 3)))
        t2 = torch.from_numpy(np.asarray(targets, dtype=np.int64).reshape((2, 1, 2, 3)))

        loss = trw.train.LossDiceMulticlass(normalization_fn=None, eps=1e-7, smooth=0, discard_background_loss=False)
        l = loss(t1, t2)
        assert len(l.shape) == 1
        assert l.shape[0] == 2

        sample_0_num_intersection_0 = 1
        sample_0_num_intersection_1 = 2

        sample_1_num_intersection_0 = 2
        sample_1_num_intersection_1 = 2

        sample_0_num_output_0 = 3
        sample_0_num_output_1 = 3

        sample_1_num_output_0 = 4
        sample_1_num_output_1 = 2

        sample_0_num_target_0 = 2
        sample_0_num_target_1 = 4

        sample_1_num_target_0 = 2
        sample_1_num_target_1 = 4

        sample_0_expected_dice_0 = 2 * sample_0_num_intersection_0 / (sample_0_num_output_0 + sample_0_num_target_0)
        sample_0_expected_dice_1 = 2 * sample_0_num_intersection_1 / (sample_0_num_output_1 + sample_0_num_target_1)

        sample_1_expected_dice_0 = 2 * sample_1_num_intersection_0 / (sample_1_num_output_0 + sample_1_num_target_0)
        sample_1_expected_dice_1 = 2 * sample_1_num_intersection_1 / (sample_1_num_output_1 + sample_1_num_target_1)

        # transform this to a loss that we want to minimize
        samples_0_expected_loss = 1.0 - (sample_0_expected_dice_0 + sample_0_expected_dice_1) / 2.0
        samples_1_expected_loss = 1.0 - (sample_1_expected_dice_0 + sample_1_expected_dice_1) / 2.0

        assert abs(samples_0_expected_loss - float(l[0])) < 1e-4
        assert abs(samples_1_expected_loss - float(l[1])) < 1e-4

    def test_losses_dice_single_multichannel_sample(self):
        found_channel0 = [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1, 1],
        ]

        found_channel1 = [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]

        found_channel2 = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
        ]

        targets = [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [2, 2, 2, 1, 1, 1],
        ]

        t1 = torch.from_numpy(np.asarray([found_channel0, found_channel1, found_channel2], dtype=np.float32).reshape((1, 3, 5, 6)))
        t2 = torch.from_numpy(np.asarray(targets, dtype=np.int64).reshape((1, 1, 5, 6)))

        loss = trw.train.LossDiceMulticlass(normalization_fn=None, eps=1e-7, smooth=0, discard_background_loss=False)
        l = loss(t1, t2)

        # calculate the mask intersection
        # we must have a mask for background & foreground
        num_intersection_0 = 3
        num_intersection_1 = 6
        num_intersection_2 = 2

        # calculate the number of pixels ON of segmentation of background & foreground
        num_output_0 = 16
        num_output_1 = 12
        num_output_2 = 2

        num_target_0 = 9
        num_target_1 = 18
        num_target_2 = 3

        # calculate the expected dice score
        expected_dice_0 = 2 * num_intersection_0 / (num_output_0 + num_target_0)
        expected_dice_1 = 2 * num_intersection_1 / (num_output_1 + num_target_1)
        expected_dice_2 = 2 * num_intersection_2 / (num_output_2 + num_target_2)

        # transform this to a loss that we want to minimize
        expected_loss = 1.0 - (expected_dice_0 + expected_dice_1 + expected_dice_2) / 3.0

        assert abs(expected_loss - float(l)) < 1e-4

    def test_dice_by_class_perfect(self):
        found_channel0 = [
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

        found_channel1 = [
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
        ]

        found_channel2 = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ]

        targets = [
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 1],
            [2, 2, 2],
        ]

        t1 = torch.from_numpy(np.asarray([found_channel0, found_channel1, found_channel2], dtype=np.float32).reshape((1, 3, 4, 3)))
        t2 = torch.from_numpy(np.asarray(targets, dtype=np.int64).reshape((1, 1, 4, 3)))

        loss = trw.train.LossDiceMulticlass(normalization_fn=None, return_dice_by_class=True, smooth=0, eps=1e-7)
        numerator, cardinality = loss(t1, t2)
        l = (numerator / cardinality).numpy().sum()
        assert abs(float(l.sum()) - 3) < 1e-5

    def test_dice_by_class_2samples(self):
        e1_found_channel0 = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 0, 0],
        ]

        e1_found_channel1 = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 1],
        ]

        e2_found_channel0 = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
        ]

        e2_found_channel1 = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ]

        targets = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ]

        t1 = torch.from_numpy(np.asarray([e1_found_channel0, e1_found_channel1, e2_found_channel0, e2_found_channel1], dtype=np.float32).reshape((2, 2, 4, 3)))
        t2 = torch.from_numpy(np.asarray([targets, targets], dtype=np.int64)).unsqueeze(1)

        loss = trw.train.LossDiceMulticlass(normalization_fn=None, return_dice_by_class=True, smooth=0, eps=0)
        numerator, cardinality = loss(t1, t2)
        dice_by_class_by_sample = (numerator / cardinality).numpy()
        average_dice_by_class = dice_by_class_by_sample.mean(axis=0)

        expected_dice_class0 = [2 * 9 / (9 + 10), 2 * 9 / (9 + 9)]
        expected_dice_class1 = [2 * 2 / (2 + 3), 2 * 3 / (3 + 3)]
        assert abs(average_dice_by_class[0] - (sum(expected_dice_class0) / 2)) < 1e-5
        assert abs(average_dice_by_class[1] - (sum(expected_dice_class1) / 2)) < 1e-5

        # test jointly the metric
        output = {
            'output_truth': t2,
            'output_raw': t1,
            'output': t1.argmax(dim=1, keepdim=True),
        }
        metric = MetricSegmentationDice(dice_fn=trw.train.LossDiceMulticlass(normalization_fn=None, return_dice_by_class=True, smooth=0, eps=0))
        metric_values = metric(output)

        assert np.abs(metric_values['numerator'].sum(axis=0) - numerator.numpy().sum(axis=0)).max() < 1e-4
        assert np.abs(metric_values['cardinality'].sum(axis=0) - cardinality.numpy().sum(axis=0)).max() < 1e-4

        # test aggregated batches
        aggregated_metric_values = metric.aggregate_metrics([metric_values, metric_values])

        one_minus_dice_class_0 = float(1 - sum(numerator[:, 0]) / sum(cardinality[:, 0]))
        one_minus_dice_class_1 = float(1 - sum(numerator[:, 1]) / sum(cardinality[:, 1]))
        one_minus_dice = float(one_minus_dice_class_0 + one_minus_dice_class_1) / 2.0

        assert abs(one_minus_dice - aggregated_metric_values['1-dice']) < 1e-5
        assert abs(one_minus_dice_class_0 - aggregated_metric_values['1-dice[class=0]']) < 1e-5
        assert abs(one_minus_dice_class_1 - aggregated_metric_values['1-dice[class=1]']) < 1e-5

    def test_dice_by_uid(self):
        # randomly generate background/foreground & truth
        # make sure analysis by sub-volume lead to the same dice as
        # full volume
        def generate_foreground(size=32):
            foreground = np.zeros([1, 1, size, size], dtype=np.int64)
            min_bb = np.random.randint(0, size - 6, size=[2], dtype=np.int64)
            max_bb = np.random.randint(6, size, size=[2], dtype=np.int64)
            foreground[0, 0, min_bb[0]:max_bb[0], min_bb[0]:max_bb[0]] = 1
            return torch.from_numpy(foreground)

        np.random.seed(1)
        nb_samples = 10

        # calculate the dice on whole data
        truth_output = []
        metric = MetricSegmentationDice(
            dice_fn=trw.train.LossDiceMulticlass(normalization_fn=None, return_dice_by_class=True, smooth=0, eps=1e-5))
        metric_intermediates = []
        for n in range(nb_samples):
            truth = generate_foreground()
            foreground = generate_foreground()
            output = torch.cat([1 - foreground.float(), foreground.float()], dim=1)
            truth_output.append((truth, output))
            i = metric({'output_truth': truth, 'output_raw': output, 'output': output.argmax(dim=1, keepdim=True)})
            metric_intermediates.append(i)
        metric_result_full_data = metric.aggregate_metrics(metric_intermediates)

        # now recalculate using sub-data blocks. If we calculate the dice by UID, we MUST find the
        # same results
        metric_by_uid = MetricSegmentationDice(
            dice_fn=trw.train.LossDiceMulticlass(normalization_fn=None, return_dice_by_class=True, smooth=0, eps=1e-5))
        metric_by_uid_intermediates = []
        bloc_size = 8
        for uid, (truth, output) in enumerate(truth_output):
            nb_blocs = np.asarray(truth.shape[2:]) // bloc_size
            for dy in range(nb_blocs[0]):
                for dx in range(nb_blocs[1]):
                    min_bb = np.asarray([dy, dx]) * bloc_size
                    max_bb = min_bb + bloc_size
                    sub_output = output[:, :, min_bb[0]:max_bb[0], min_bb[1]:max_bb[1]]
                    sub_truth = truth[:, :, min_bb[0]:max_bb[0], min_bb[1]:max_bb[1]]
                    i = metric_by_uid({'output_truth': sub_truth, 'output_raw': sub_output, 'output': sub_output.argmax(dim=1, keepdim=True), 'uid': [uid]})
                    metric_by_uid_intermediates.append(i)

        metric_result_sub = metric.aggregate_metrics(metric_by_uid_intermediates)

        # should get the same results!
        for name, value in metric_result_full_data.items():
            value_sub = metric_result_sub[name]
            assert abs(value_sub - value) < 1e-6
            print(value_sub - value)

    def test_focal_loss_binary__multiclass_id(self):
        # in this configuration, the cross entropy loss and the focal loss MUST be identical
        # using the multi-class formulation (C=2)
        targets = torch.from_numpy(np.asarray([0, 1, 1, 0, 1], dtype=np.int64))
        outputs = torch.from_numpy(np.asarray([[0.5, 0.1], [0.1, 0.9], [0.1, 0.8], [0.6, 0.1], [0.3, 0.6]], dtype=np.float32))
        alpha = np.asarray([1.0, 1.0], dtype=np.float32)

        loss_focal_fn = trw.train.LossFocalMulticlass(alpha=alpha, gamma=0.0)
        loss_focal = loss_focal_fn(outputs, targets.unsqueeze(dim=1))
        assert loss_focal.shape == (5,)

        loss_ce_fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss_ce = loss_ce_fn(outputs, targets)

        # gamma == 0 means we MUST retrun cross entropy
        assert float((loss_focal - loss_ce).abs().max()) <= 1e-6

    def test_focal_loss_binary_id(self):
        # in this configuration, the cross entropy loss and the focal loss MUST be identical
        # using the binary formulation (C=1)
        targets = torch.from_numpy(np.asarray([0, 1, 1, 0, 1], dtype=np.int64))
        outputs = torch.from_numpy(np.asarray([0.4, 0.9, 0.8, 0.4, 0.6], dtype=np.float32)).unsqueeze(1)

        loss_focal_fn = trw.train.LossFocalMulticlass(gamma=0.0)
        loss_focal = loss_focal_fn(outputs, targets.unsqueeze(dim=1))
        assert loss_focal.shape == (5,)

        loss_ce = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets.unsqueeze(1).float(), reduction='none')

        # gamma == 0 means we MUST retrun cross entropy
        assert float((loss_focal.squeeze() - loss_ce.squeeze()).abs().max()) <= 1e-6

    def test_focal_loss_binary_ordering(self):
        # make sure the more confident classes get less loss weigth
        targets = torch.from_numpy(np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64))
        outputs = torch.from_numpy(np.asarray([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]], dtype=np.float32))
        alpha = torch.from_numpy(np.asarray([1.0, 1.0], dtype=np.float32))

        loss_focal_fn = trw.train.LossFocalMulticlass(alpha=alpha, gamma=10.0)
        loss_focal = trw.utils.to_value(loss_focal_fn(outputs, targets.unsqueeze(dim=1)))
        assert len(loss_focal) == len(targets)

        ce_loss = trw.utils.to_value(torch.nn.CrossEntropyLoss(reduction='none')(outputs, targets))

        # rescale the focal loss to have a similar max range
        loss_focal_scaled = max(ce_loss) / max(loss_focal) * loss_focal

        assert loss_focal_scaled[0] * 10 < ce_loss[0]
        assert loss_focal_scaled[5] * 10 < ce_loss[5]

        assert loss_focal_scaled[1] * 3 < ce_loss[1]
        assert loss_focal_scaled[4] * 3 < ce_loss[4]

    def test_focal_loss_multiclass_2d(self):
        targets = torch.ones([10, 5, 5], dtype=torch.int64)
        outputs = torch.ones([10, 3, 5, 5], dtype=torch.float32)

        loss_focal_fn = trw.train.LossFocalMulticlass(gamma=1.0)
        loss_focal = trw.utils.to_value(loss_focal_fn(outputs, targets.unsqueeze(dim=1)))
        assert len(loss_focal) == len(targets)

    def test_triplet_loss(self):
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
                [[[1, 3.1]]],   # not satisfying the margin
                [[[1, 40.1]]],  # far from the samples_p and margin, should be 0 loss
            ], dtype=torch.float32)

        loss = trw.train.LossTriplets(margin=0.5)
        losses = loss(samples, samples_p, samples_n)
        assert (losses == torch.tensor([0, 0.5, 0])).all()

    def test_center_loss(self):
        # make sure the per-class centers are fitted to the data
        features = torch.tensor(
            [
                [[[1, 2]]],  # make sure it works for N-d features
                [[[1, 3]]],
                [[[2, 14]]],
                [[[2, 15]]],
            ], dtype=torch.float32)
        classes = torch.tensor([0, 0, 1, 1])

        nb_iter = 1000
        model = IdNetCenterLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

        for i in range(nb_iter):
            batch = {'features': features, 'classes': classes}
            o = model(batch)
            optimizer.zero_grad()
            loss = o.evaluate_batch(batch, True)['loss']
            loss.backward()
            optimizer.step()

        centers = model.center_loss.centers
        expected_centers = torch.tensor(
            [
                [1, 2.5],
                [2, 14.5],
            ], dtype=torch.float32
        )

        assert (expected_centers - centers).abs().sum() < 1e-3

    def test_contrastive_loss(self):
        # dissimilar pairs outside radius contributes to 0 loss
        x0 = torch.tensor([[0], [-1]], dtype=torch.float32)
        x1 = torch.tensor([[5], [-5]], dtype=torch.float32)
        losses = LossContrastive(margin=3)(x0, x1, torch.zeros([2]))
        assert len(losses) == 2
        assert (losses <= 0).all()

        # for similar pairs, loss is the distance between the samples
        x0 = torch.tensor([[1], [2]], dtype=torch.float32)
        x1 = torch.tensor([[2], [4]], dtype=torch.float32)
        losses = LossContrastive(margin=1)(x0, x1, torch.ones([2]))
        assert len(losses) == 2

        expected_loss = (x0 - x1) * (x0 - x1) * 0.5
        assert (expected_loss.squeeze() - losses).abs().max() <= 1e-3

    def test_total_variation_2d(self):
        x = torch.tensor([[1, 3, 0], [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # must be in NCHW
        tv = trw.train.total_variation_norm(x, 2)
        expected = ((1 - 0) ** 2 + (3 - 0) ** 2) / 3 + ((1 - 3) ** 2 + (3 - 0) ** 2) / (2 * (3 - 1))
        assert abs(tv - expected) < 1e-5

    def test_total_variation_3d(self):
        x = torch.ones([10, 1, 5, 6, 7])
        tv = trw.train.total_variation_norm(x, 2)
        # no variation, so expect a 0 loss
        assert abs(tv - 0) < 1e-5

    def test_loss_ce_csi(self):
        loss_fn = LossCrossEntropyCsiMulticlass()
        targets = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
        x = torch.tensor([
            [0.1, 0, 0],  # no loss, TN
            [0, 0.1, 0],  # keep loss, important class
            [0, 0, 0.1],  # no loss, TN
            [0, 0, 0.1],  # loss, error!
            [0, 0, 0.1],  # loss, error!
        ], dtype=torch.float32)

        losses = loss_fn(x, targets)
        assert losses[0] == 0
        assert losses[1] > 0
        assert losses[2] == 0
        assert losses[3] > 0
        assert losses[4] > 0

    def test_loss_f1(self):
        nb_samples = 100
        np.random.seed(1)

        for exp in range(100):
            results = []
            for n in range(200):
                r = np.random.uniform(0, 1, size=[nb_samples]).astype(np.float32)
                output = np.zeros([nb_samples, 2], dtype=np.float32)
                output[:, 0] = r
                output[:, 1] = 1 - r

                output_values = torch.from_numpy(output).type(torch.float32)
                truth_values = torch.from_numpy((np.random.uniform(0, 1, size=[nb_samples]) >= 0.7).astype(np.int64))

                loss = trw.train.LossBinaryF1()
                surrogate_f1 = loss(output_values, truth_values)
                assert len(surrogate_f1) == nb_samples
                surrogate_f1 = surrogate_f1.mean()
                f1 = 1 - sklearn.metrics.f1_score(truth_values, output.argmax(axis=1), average='macro')
                results.append((f1, float(surrogate_f1)))

            d = np.asarray(results)
            r, _ = scipy.stats.pearsonr(d[:, 0], d[:, 1])
            print('r=', r)
            assert r > 0.7  # we MUST have a good correlation!

            # from matplotlib import pyplot as plt
            # plt.scatter(d[:, 0], d[:, 1])
            # plt.ylabel('Macro F1-score')
            # plt.xlabel('Differentiable F1 loss')
            # plt.show()

    def test_mse_packed(self):
        metric = trw.train.LossMsePacked(reduction='none')

        targets = torch.randint(0, 3, [10, 1, 5, 6], dtype=torch.long)
        outputs = torch.randint(0, 1, [10, 3, 5, 6], dtype=torch.float32)
        losses = metric(outputs, targets)

        expected_loss = (trw.train.one_hot(targets.squeeze(1), 3) - outputs) ** 2
        assert (losses - expected_loss).abs().max() < 1e-5

    def test_mse_packed_binary(self):
        """
        Test output of a binary classifier
        """
        metric = trw.train.LossMsePacked(reduction='none')

        targets = torch.randint(0, 1, [10, 1, 5, 6], dtype=torch.long)
        outputs = torch.randint(0, 1, [10, 1, 5, 6], dtype=torch.float32)
        losses = metric(outputs, targets)

        expected_loss = (targets.type(torch.float32) - outputs) ** 2
        assert (losses - expected_loss).abs().max() < 1e-5

    @staticmethod
    def jaccard_loss(true, logits, eps=1e-7):
        """Computes the Jaccard loss, a.k.a the IoU loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the jaccard loss so we
        return the negated jaccard loss.
        Args:
            true: a tensor of shape [B, H, W] or [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            jacc_loss: the Jaccard loss.
        """
        from torch.nn import functional as F
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        union = cardinality - intersection
        jacc_loss = (intersection / (union + eps)).mean()
        return (1 - jacc_loss)
        
    def test_jaccard_loss(self):
        found_channel0 = [
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

        found_channel1 = [
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
        ]

        found_channel2 = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ]

        targets = [
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 1],
            [2, 2, 2],
        ]

        t1 = torch.from_numpy(np.asarray([found_channel0, found_channel1, found_channel2], dtype=np.float32).reshape((1, 3, 4, 3)))
        t2 = torch.from_numpy(np.asarray(targets, dtype=np.int64).reshape((1, 1, 4, 3)))

        truth_loss = self.jaccard_loss(t2, t1)
        loss = trw.train.LossJaccardMulticlass(normalization_fn=None, smooth=0, discard_background_loss=False)