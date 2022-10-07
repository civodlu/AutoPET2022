from .simple_layers import SimpleModule
from ..layers import SubTensor as SubTensor_layers



class SubTensor(SimpleModule):
    """
    Select a region of a tensor (without copy), excluded the first component (N)
    """
    def __init__(self, node, min_indices, max_indices_exclusive):
        """
        Args:
            node: the parent node
            min_indices: the minimum indices to select for each dimension, excluded the first component (N)
            max_indices_exclusive: the maximum indices (excluded) to select for each dimension,
                excluded the first component (N)
        """
        super().__init__(node=node, module=SubTensor_layers(min_indices, max_indices_exclusive))
