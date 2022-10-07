from unittest import TestCase
import torch
import trw
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleNet_1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        x1 = batch['x1']
        x2 = batch['x2']

        z1 = F.relu(x1)
        z2 = F.relu(x2)

        return {
            'output_1': trw.train.OutputEmbedding(F.relu(z1 - 1 - z2), clean_loss_term_each_batch=True)
        }


class SimpleNet_2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        x1 = batch['x1']
        x2 = batch['x2']

        z1 = F.relu(x1 - 1)
        z2 = F.relu(x2)

        return {
            'output_1': trw.train.OutputEmbedding(F.relu(z1 - z2))
        }


class TestIntegratedGradients(TestCase):
    @staticmethod
    def calculate_attribution(model):
        # test attribution counter example in the paper "Axiomatic Attribution for Deep Networks"
        # Mukund Sundararajan, Ankur Taly, Qiqi Yan
        # Supplemental materials, Section B

        batch = {
            'x1': torch.from_numpy(np.asarray([[3]], dtype=np.float32)),
            'x2': torch.from_numpy(np.asarray([[1]], dtype=np.float32))
        }
        batch['x1'].requires_grad = True
        batch['x2'].requires_grad = True

        explainer = trw.train.IntegratedGradients(model=model, use_output_as_target=True, steps=2000)
        output_name, attributions = explainer(batch, target_class_name='output_1')
        assert output_name == 'output_1'
        assert abs(1.5 - float(attributions['x1'])) < 0.01
        assert abs(-0.5 - float(attributions['x2'])) < 0.01

    def test_integrated_relu(self):
        # test attribution from very simple networks
        model = SimpleNet_1()
        TestIntegratedGradients.calculate_attribution(model)

        model = SimpleNet_2()
        TestIntegratedGradients.calculate_attribution(model)
