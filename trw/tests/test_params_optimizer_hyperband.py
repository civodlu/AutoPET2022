from unittest import TestCase
import trw.hparams
import trw.train
import numpy as np
import os


class TestParamsOptimizer(TestCase):
    def test_search(self):
        #
        # very simple test just to test the API: minimize x, y given the loss l(x, y) = x^2 + y^2. Best params: x, y = (0, 0)
        # TODO: this is not assessing the perforamce of hyperband vs random search, but we should do that too!
        #

        def evaluate_hparams(hparams, iterations):
            x = hparams.create(trw.hparams.ContinuousUniform('param_x', 5, -10, 10))
            y = hparams.create(trw.hparams.ContinuousUniform('param_y', 5, -10, 10))
            d = hparams.create(trw.hparams.ContinuousUniform('param_d', 5, -10, 10))

            return {'loss': x * x + y * y + d}, [], {'param_x': x, 'param_y': y, 'param_d': d}

        optimizer = trw.hparams.params_optimizer_hyperband.HyperParametersOptimizerHyperband(
            loss_fn=lambda metrics: metrics['loss'],
            evaluate_fn=evaluate_hparams,
            max_iter=81,
            repeat=20)

        options = trw.train.Options()
        tmp = os.path.join(options.workflow_options.logging_directory, 'hyperband_test')
        trw.train.create_or_recreate_folder(tmp)
        tries = optimizer.optimize(store=None)

        values = [t.metrics['loss'] for t in tries]
        best_try = tries[np.argmin(np.abs(values))]
        print(str(best_try))
        self.assertTrue(best_try.metrics['loss'] < 1.0)
