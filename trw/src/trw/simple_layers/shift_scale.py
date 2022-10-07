from .simple_layers import SimpleModule
from ..layers import ShiftScale as ShiftScale_layers


class ShiftScale(SimpleModule):
    """
    Normalize a tensor with a mean and standard deviation

    The output tensor will be (x - mean) / standard_deviation

    This layer simplify the preprocessing for the `trw.simple_layers` package
    """
    def __init__(self, node, mean, standard_deviation):
        """
        Args:
            node: the input node
            mean: the mean. Must be broadcastable to node shape
            standard_deviation: the standard deviation. Must be broadcastable to node shape
        """
        super().__init__(node=node, module=ShiftScale_layers(mean, standard_deviation), shape=node.shape)


