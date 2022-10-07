from typing import Callable, Tuple, Any, Optional, List

from ..basic_typing import History
from .params_optimizer import HyperParametersOptimizer
from .store import Metrics, RunStore, RunResult
from .params import HyperParameters, HyperParameterRepository

import copy
import math
import numpy as np
import logging


logger = logging.getLogger(__name__)


def log_hyperband(msg):
    logger.info(msg)


class HyperParametersOptimizerHyperband(HyperParametersOptimizer):
    """
    Implementation of `Hyperband: a novel bandit based approach to hyper-parameter optimization` [#]_

    .. [#] https://arxiv.org/abs/1603.06560
    """
    def __init__(self, evaluate_fn: Callable[[HyperParameters, float], Tuple[Metrics, History, Any]],
                 loss_fn: Callable[[Metrics], float],
                 max_iter: int = 81,
                 eta: int = 3,
                 repeat: int = 100,
                 log_string: Callable[[str], None] = log_hyperband,
                 always_include_default_hparams_in_each_cycle: bool = True):
        """
        Table of runs for max_iter=81, eta=3

        max_iter = 81        s=4             s=3             s=2             s=1             s=0
        eta = 3              n_i   r_i       n_i   r_i       n_i   r_i       n_i   r_i       n_i   r_i
        B = 5*max_iter       ---------       ---------       ---------       ---------       ---------
                              81    1         27    3         9     9         6     27        5     81
                              27    3         9     9         3     27        2     81
                              9     9         3     27        1     81
                              3     27        1     81
                              1     81

        n_i = number of configurations
        r_i = number of iteration per configuration

        Args:
            evaluate_fn: evaluation function, returning metrics, history and info
            max_iter: the maximum number of epoch for the training of the best configuration
            eta: downsampling rate
            repeat: number of times hyperband will be repeated
            log_string: defines how to report results
            always_include_default_hparams_in_each_cycle: if True, for each outer loop,
                the default parameters are evaluated for the first repeat only!
            loss_fn: extract a loss to minimize from the metrics
        """
        self.evaluate_fn = evaluate_fn
        self.log_string = log_string
        self.max_iter = max_iter
        self.eta = eta
        self.repeat = repeat
        self.loss_fn = loss_fn
        self.always_include_default_hparams_in_each_cycle = always_include_default_hparams_in_each_cycle

    def _repeat_one(
            self,
            repeat_id,
            nb_runs,
            hyper_parameters: HyperParameters,
            store: Optional[RunStore] = None) -> Tuple[List[RunResult], int]:
        """
        Run full Hyperband search

        Args:
            repeat_id: the iteration number
            nb_runs: the run number
            store: how to store the result
            hyper_parameters: the hyper-parameters

        Returns:
            a tuple of list of runs and number of runs for this iteration og hyperband
        """

        log = math.log
        default_parameters = copy.deepcopy(hyper_parameters)

        self.log_string(f'repeat_id={repeat_id}, run={nb_runs}')

        logeta = lambda x: log(x) / log(self.eta)

        # number of unique executions of Successive Halving (minus one)
        s_max = int(logeta(self.max_iter))

        # total number of iterations (without reuse) per execution of Succesive Halving (n,r)
        B = (s_max + 1) * self.max_iter

        #
        # Code adapted from https://homes.cs.washington.edu/~jamieson/hyperband.html
        #

        # Begin Finite Horizon Hyperband outlerloop. Repeat indefinitely.
        results_last = []
        for s in reversed(range(s_max + 1)):
            n = int(math.ceil(int(B / self.max_iter / (s + 1)) * self.eta ** s))  # initial number of configurations
            r = self.max_iter * self.eta ** (-s)  # initial number of iterations to run configurations for

            # Begin Finite Horizon Successive Halving with (n,r)
            T = []
            for i in range(n):
                if self.always_include_default_hparams_in_each_cycle and i == 0 and repeat_id == 0:
                    # include the default configuration for the 1 repeat
                    T.append(copy.deepcopy(default_parameters))
                else:
                    hyper_parameters.randomize()
                    T.append(copy.deepcopy(hyper_parameters))

            run_params = None
            for i in range(s + 1):
                # Run each of the n_i configs for r_i iterations and keep best n_i/eta
                n_i = n * self.eta ** (-i)
                r_i = r * self.eta ** (i)

                val_results = []
                val_losses = []
                for t_index, t in enumerate(T):
                    metrics, history, info = self.evaluate_fn(t, r_i)
                    loss = self.loss_fn(metrics)
                    metrics['hparams_loss'] = loss
                    self.log_string(f'run={nb_runs}, s={s}, r_i={r_i}, loss={loss}, params={t.hparams}, info={info}')

                    run_result = RunResult(
                        metrics=metrics,
                        history=history,
                        info={
                            'run': nb_runs,
                            's': s,
                            'r_i': r_i,
                            'info': info
                        },
                        hyper_parameters=copy.deepcopy(t)
                    )

                    val_results.append(run_result)
                    val_losses.append(loss)

                    if store is not None:
                        store.save_run(run_result=run_result)

                    nb_runs += 1

                best_runs = np.argsort(val_losses)[0:int(n_i / self.eta)]
                run_params = []
                for run in range(len(val_losses)):
                    run_params.append(val_results[run])

                T = [T[i] for i in best_runs]

            assert run_params is not None, 'implementation is wrong!'
            results_last += run_params  # keep track of the last round for each `s`
        return results_last, nb_runs

    def optimize(self, store: Optional[RunStore], hyper_parameters: Optional[HyperParameters] = None) -> List[RunResult]:
        """
        Optimize the hyper parameters using Hyperband
        
        Args:
            store: how to result of each run. Can be None, in this case nothing is exported.
            hyper_parameters: the hyper parameters to be optimized. If `None`,
                use the global repository :class:`trw.hparams.HyperParameterRepository`

        Returns:
            the results of all the runs
        """
        if hyper_parameters is None:
            # do not create a new store instance: the model might have
            # instantiated hyper-parameters already, before the `evaluate_fn`
            # so instead, use existing repository
            self.log_string('creating an empty hyper parameter repository')
            hyper_parameters = HyperParameterRepository.current_hparams
        else:
            # update the global repository
            HyperParameterRepository.current_hparams = hyper_parameters

        # here `discover` the hyper-parameter. We must assume the hyper-parameter list
        # won't change
        self.evaluate_fn(hyper_parameters, 1)

        results = []
        nb_runs = 0
        for repeat_id in range(self.repeat):
            r, nb_runs = self._repeat_one(store=store,
                                          hyper_parameters=hyper_parameters,
                                          repeat_id=repeat_id,
                                          nb_runs=nb_runs)
            results += r
        return results
