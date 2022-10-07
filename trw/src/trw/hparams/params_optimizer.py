from abc import ABC, abstractmethod
from typing import Optional, List

from .store import RunStore, RunResult


class HyperParametersOptimizer(ABC):
    @abstractmethod
    def optimize(self, store: Optional[RunStore]) -> List[RunResult]:
        pass
