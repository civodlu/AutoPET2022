from typing import Tuple, Optional, Any, Callable

import copy
import logging

from ..basic_typing import History
from .params_optimizer import HyperParametersOptimizer
from .store import Metrics, RunStore, RunResult
from .params import HyperParameters, HyperParameterRepository
from ..utils import ExceptionAbortRun

logger = logging.getLogger(__name__)


def log_with_logger(string):
    """
    Log string to console
    """
    logger.info(string)


class HyperParametersOptimizerRandomSearchLocal(HyperParametersOptimizer):
    """
    hyper-parameter search using a random walk on a single machine
    """
    def __init__(self,
                 evaluate_fn: Callable[[HyperParameters], Tuple[Metrics, History, Any]],
                 repeat: int,
                 log_string: Callable[[str], None] = log_with_logger):
        """
        
        Args:
            evaluate_fn: th evaluation function taking as input hyper-parameters and returning
                a tuple (metrics, history, evaluate info). `evaluate_fn` may raise :class:`ExceptionAbortRun`
                to early terminate a run
            repeat: the number of random iterations
            log_string: how to log the search
        """
        self.evaluate_fn = evaluate_fn
        self.repeat = repeat
        self.log_string = log_string

    def optimize(self, store: Optional[RunStore] = None, hyper_parameters: Optional[HyperParameters] = None):
        """
        Optimize the hyper-parameter search using random search
        
        Args:
            store: defines how the runs will be saved
            hyper_parameters: the hyper parameters to be optimized. If `None`,
                use the global repository :class:`trw.hparams.HyperParameterRepository`
        """
        self.log_string('started optimization')
        if hyper_parameters is None:
            # do not create a new store instance: the model might have
            # instantiated hyper-parameters already, before the `evaluate_fn`
            # so instead, use existing repository
            self.log_string('creating an empty hyper parameter repository')
            hyper_parameters = HyperParameterRepository.current_hparams
        else:
            # update the global repository
            HyperParameterRepository.current_hparams = hyper_parameters

        self.log_string(f'started optimize(), hyper_parameters={hyper_parameters}')

        # then randomly evaluate random hyper parameters and export the results
        iteration = 0
        results = []
        while self.repeat != iteration:
            # by default, evaluate the given hyper parameter default value
            try:
                status = 'SUCCESS'
                metrics, history, info = self.evaluate_fn(hyper_parameters)
            except ExceptionAbortRun as e:
                # the run was aborted early. Record only the history of the run
                metrics = e.metrics
                history = e.history
                info = e.reason
                status = 'FAILED'
                self.log_string(f'iteration={iteration} was terminated early (epoch={len(history)}). Reason={e.reason}')
            except RuntimeError as e:
                metrics = None  # type: ignore
                history = None  # type: ignore
                status = 'FAILED'
                info = f'Run FAILED. Exception={e}'
                self.log_string(f'iteration={iteration} FAILED. Exception={e}')

            if iteration == 0:
                self.log_string(f'hyper_parameters (first run)={hyper_parameters}')
                
            if status == 'SUCCESS':
                self.log_string(f'status={status} iteration={iteration}, metrics={metrics}, '
                                f'params={str(hyper_parameters.hparams)}')

            run_result = RunResult(
                metrics=metrics,
                hyper_parameters=copy.deepcopy(hyper_parameters),
                history=history,
                info=info
            )
            if store is not None:
                store.save_run(run_result)

            results.append(run_result)
            iteration += 1

            if len(hyper_parameters) == 0:
                self.log_string('No hyper-parameter found. Random search stopped!')
                break

            # at the end of the iteration, randomly modify the hyper parameters
            hyper_parameters.randomize()

        self.log_string('hyper-parameter optimization completed!')
        return results
