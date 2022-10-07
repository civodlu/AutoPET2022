import trw
from unittest import TestCase
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 2)
        self.fc1 = nn.Linear(20 * 6 * 6, 500)
        self.fc2 = nn.Linear(500, 10)
        self.relu_fc1 = nn.ReLU()
        self.relu_conv1 = nn.ReLU()

    def forward(self, batch):
        # a batch should be a dictionary of features
        x = batch['images'] / 255.0

        x = self.relu_conv1(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 20 * 6 * 6)
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)

        # here we create a softmax output that will use
        # the `targets` feature as classification target
        return x


class Net2inputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 2)
        self.fc1 = nn.Linear(20 * 6 * 6, 500)
        self.fc2 = nn.Linear(500, 10)
        self.relu_fc1 = nn.ReLU()
        self.relu_conv1 = nn.ReLU()

    def forward(self, batch):
        # a batch should be a dictionary of features
        x = batch['images'] / 255.0
        x2 = batch['other']

        x = self.relu_conv1(self.conv1(x))
        x_single = F.max_pool2d(x, 2, 2)
        x = x_single.view(-1, 20 * 6 * 6) + x2
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)

        # here we create a softmax output that will use
        # the `targets` feature as classification target
        return x, x_single


class TestFindTensorLeaves(TestCase):
    def test_single_input(self):
        """
        We have a simple neural net, make sure we can discover its input
        """
        i = torch.zeros([10, 1, 28, 28], requires_grad=True)
        model = Net()
        model.eval()
        output = model({'images': i})

        leaves = trw.train.find_tensor_leaves_with_grad(output)
        assert len(leaves) == 1
        assert leaves[0] is i

    def test_non_discoverable_input(self):
        """
        The input will not have its grad computed, so we can't use autograd mechanism to discover it
        """
        i = torch.zeros([10, 1, 28, 28], requires_grad=False)
        model = Net()
        model.eval()
        output = model({'images': i})

        leaves = trw.train.find_tensor_leaves_with_grad(output)
        assert len(leaves) == 0

    def test_two_input(self):
        """
        Neural net with 2 heads
        """
        i = torch.zeros([10, 1, 28, 28], requires_grad=True)
        i2 = torch.zeros([10, 720], requires_grad=True)
        model = Net2inputs()
        model.eval()
        output, output_single = model({'images': i, 'other': i2})

        leaves = trw.train.find_tensor_leaves_with_grad(output)
        assert len(leaves) == 2
        assert leaves[0] is i2
        assert leaves[1] is i

        leaves = trw.train.find_tensor_leaves_with_grad(output_single)
        assert len(leaves) == 1
        assert leaves[0] is i

    def test_last_convolution_simple(self):
        last = nn.Conv2d(3, 4, 5)

        model = nn.Sequential(
            nn.Conv2d(1, 2, 5),
            nn.ReLU(),
            nn.Conv2d(2, 3, 5),
            nn.ReLU(),
            last,
            nn.ReLU(),
            nn.Dropout2d(),
        )

        inputs = torch.zeros([1, 1, 32, 32], requires_grad=True)
        r = trw.train.find_last_forward_convolution(model, inputs)
        assert r is not None
        m = r['matched_module']
        i = r['matched_module_inputs']
        o = r['matched_module_output']
        outputs = r['outputs']

        assert m is last
        assert i is not None
        assert o is not None
        assert outputs is not None

    def test_last_convolution_nested(self):
        """
        Make sure we go through each module recursively
        """
        last = nn.Conv2d(3, 4, 5)

        model = nn.Sequential(
            nn.Conv2d(1, 2, 5),
            nn.ReLU(),
            nn.Conv2d(2, 3, 5),
            nn.Sequential(
                nn.ReLU(),
                last,
                nn.ReLU()
            ),
            nn.Dropout2d(),
        )

        inputs = torch.zeros([1, 1, 32, 32], requires_grad=True)
        r = trw.train.find_last_forward_convolution(model, inputs)
        assert r is not None
        m = r['matched_module']
        i = r['matched_module_inputs']
        o = r['matched_module_output']
        outputs = r['outputs']

        assert m is last
        assert i is not None
        assert o is not None
        assert outputs is not None

    def test_last_convolution_simple_relative_index(self):
        first = nn.Conv2d(1, 2, 5)
        last = nn.Conv2d(3, 4, 5)

        model = nn.Sequential(
            first,
            nn.ReLU(),
            nn.Conv2d(2, 3, 5),
            nn.ReLU(),
            last,
            nn.ReLU(),
            nn.Dropout2d(),
        )

        inputs = torch.zeros([1, 1, 32, 32], requires_grad=True)
        r = trw.train.find_last_forward_convolution(model, inputs, relative_index=2)
        assert r is not None
        m = r['matched_module']
        i = r['matched_module_inputs']
        o = r['matched_module_output']
        outputs = r['outputs']

        assert m is first
        assert i is not None
        assert o is not None
        assert outputs is not None
