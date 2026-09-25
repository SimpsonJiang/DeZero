from dezero.core import Variable, Function, Parameter
from dezero.core import using_config, no_grad, as_array, as_variable, setup_variable
from dezero.utils import plot_dot_graph
import dezero.functions as F
setup_variable()
Variable.__getitem__ = F.get_item