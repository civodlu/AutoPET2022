from unittest import TestCase
import trw
import torch


class TestCleanAddedHooks(TestCase):
    def test_tracking(self):
        """
        Make sure we can track the added hooks at the module and sub-modules level
        """
        N, D_in, H, D_out = 64, 1000, 100, 10

        linear = torch.nn.Linear(D_in, H)
        relu = torch.nn.ReLU()

        model = torch.nn.Sequential(
            linear,
            relu,
            torch.nn.Linear(H, D_out),
        )

        def hook(module, i, o):
            pass

        linear.register_backward_hook(hook)
        relu.register_forward_hook(hook)
        model.register_backward_hook(hook)

        forward, backard = trw.train.CleanAddedHooks.record_hooks(model)
        assert len(forward) == 1
        assert len(backard) == 2

    def test_restoration_from_empty(self):
        """
        Restore the hooks as original (empty)
        """
        N, D_in, H, D_out = 64, 1000, 100, 10

        linear = torch.nn.Linear(D_in, H)
        relu = torch.nn.ReLU()

        model = torch.nn.Sequential(
            linear,
            relu,
            torch.nn.Linear(H, D_out),
        )

        def hook(module, i, o):
            pass

        with trw.train.CleanAddedHooks(model):
            linear.register_backward_hook(hook)
            relu.register_forward_hook(hook)
            model.register_backward_hook(hook)

            forward, backard = trw.train.CleanAddedHooks.record_hooks(model)
            assert len(forward) == 1
            assert len(backard) == 2

        forward, backard = trw.train.CleanAddedHooks.record_hooks(model)
        assert len(forward) == 0
        assert len(backard) == 0

    def test_restoration_from_non_empty(self):
        """
        Restore the hooks as original (2 forward and 2 backward pre-existing hooks)
        """
        N, D_in, H, D_out = 64, 1000, 100, 10

        linear = torch.nn.Linear(D_in, H)
        relu = torch.nn.ReLU()

        model = torch.nn.Sequential(
            linear,
            relu,
            torch.nn.Linear(H, D_out),
        )

        def hook(module, i, o):
            pass

        def hook2(module, i, o):
            pass

        linear.register_backward_hook(hook)
        linear.register_forward_hook(hook)
        relu.register_forward_hook(hook)
        relu.register_backward_hook(hook)

        forward, backard = trw.train.CleanAddedHooks.record_hooks(model)
        assert len(forward) == 2
        assert len(backard) == 2

        with trw.train.CleanAddedHooks(model):
            model.register_backward_hook(hook2)
            model.register_forward_hook(hook2)

        forward, backard = trw.train.CleanAddedHooks.record_hooks(model)
        assert len(forward) == 2
        assert len(backard) == 2
