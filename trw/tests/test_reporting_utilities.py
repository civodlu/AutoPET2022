import trw.utils


import trw
from unittest import TestCase
import torch.nn as nn


class Sub2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 10)


class Sub1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(1, 10, 5)
        self.sub2 = Sub2()


class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub = Sub1()


class TestReportingUtilities(TestCase):
    def test_hierarchical_name(self):
        model = Root()
        d = trw.utils.collect_hierarchical_module_name('Root', model)
        assert d[model.sub.conv1] == 'Root/Sub1_0/Conv2d_0'
        assert d[model.sub.conv2] == 'Root/Sub1_0/Conv2d_1'
        assert d[model.sub.sub2.linear] == 'Root/Sub1_0/Sub2_2/Linear_0'
        assert d[model.sub.sub2] == 'Root/Sub1_0/Sub2_2'
        assert d[model.sub] == 'Root/Sub1_0'
        assert d[model] == 'Root'

    def test_hierarchical_parameter(self):
        model = Root()
        d = trw.utils.collect_hierarchical_parameter_name('Root', model, with_grad_only=True)

        module_d = dict(model.sub.conv1.named_parameters(recurse=False))
        assert d[module_d['weight']] == 'Root/Sub1_0/Conv2d_0/weight'
        assert d[module_d['bias']] == 'Root/Sub1_0/Conv2d_0/bias'

        module_d = dict(model.sub.conv2.named_parameters(recurse=False))
        assert d[module_d['weight']] == 'Root/Sub1_0/Conv2d_1/weight'
        assert d[module_d['bias']] == 'Root/Sub1_0/Conv2d_1/bias'

        module_d = dict(model.sub.sub2.linear.named_parameters(recurse=False))
        assert d[module_d['weight']] == 'Root/Sub1_0/Sub2_2/Linear_0/weight'
        assert d[module_d['bias']] == 'Root/Sub1_0/Sub2_2/Linear_0/bias'
