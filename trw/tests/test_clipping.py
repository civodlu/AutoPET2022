import unittest
import trw
import torch.nn as nn
import torch
from torch import optim
from collections import Iterable


class TestClipping(unittest.TestCase):
    def test_gradient_clipping_nograd(self):
        """
        Gradient should not be updated!
        """
        p = nn.Linear(1, 1)
        p.weight.data[0, 0] = 1.0
        

        optimizer = optim.SGD(p.parameters(), 0.1)
        optimizer_clipped = trw.train.ClippingGradientNorm(optimizer, 0.0)

        optimizer_clipped.zero_grad()
        i = torch.ones([20, 1], dtype=torch.float32)
        loss = (p.forward(i) ** 2).sum()
        loss.backward()
        assert (p.weight.grad ** 2) .sum() > 1 
        optimizer_clipped.step()

        assert abs(p.weight.data[0, 0] - 1.0) < 1e-8

    def test_gradient_clipping_small(self, model=nn.Linear(1, 1, bias=False), optimizer=None, max_norm = 2.0, learning_rate = 0.05):
        """
        Gradient should not be updated more than what is specified
        """
        w = 3.0
        model.weight.data[0, 0] = w

        if optimizer is None:
            optimizer = trw.train.ClippingGradientNorm(optim.SGD(model.parameters(), learning_rate), max_norm, norm_type=1)

        optimizer.zero_grad()
        i = torch.ones([10, 1], dtype=torch.float32)
        loss = (model.forward(i) ** 2).sum()
        loss.backward()
        assert (model.weight.grad ** 2) .sum() > 1 
        optimizer.step()

        expected_w = w - learning_rate * max_norm

        error = abs(model.weight.data[0, 0] - expected_w)
        assert error < 1e-5

    def test_gradient_clipping_small_optimizer(self):
        """
        Test the clipping is applied
        """
        model=nn.Linear(1, 1, bias=False)
        max_norm = 1.0
        learning_rate = 0.2
        optimizer_fn = trw.train.OptimizerSGD(learning_rate=learning_rate, momentum=0).clip_gradient_norm(max_norm=max_norm)
        optimizers = optimizer_fn({'dataset1': None}, model)
        self.test_gradient_clipping_small(
            model=model, 
            optimizer=optimizers[0]['dataset1'], 
            max_norm=max_norm, 
            learning_rate=learning_rate
        )


if __name__ == '__main__':
    unittest.main()