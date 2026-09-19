from dezero.core import Variable, Function, Parameter
from dezero.core import using_config, no_grad, as_array, as_variable, setup_variable
from dezero.layers import Layer, Linear
from dezero.models import Model, MLP
from dezero.utils import plot_dot_graph
setup_variable()