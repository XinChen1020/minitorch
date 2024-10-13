from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol
from collections import deque

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    xvals = list(vals)

    xvals[arg] += epsilon
    fx_plus = f(*xvals)

    xvals[arg] -= 2 * epsilon
    fx_minus = f(*xvals)

    return (fx_plus - fx_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    result = deque([])
    visited = set()

    def visit(v: Variable) -> None:
        if v.unique_id in visited or v.is_constant():
            return
        
        visited.add(v.unique_id)

        if not v.is_leaf():
            for parent in v.parents: 
                visit(parent)
                
        result.appendleft(v)
    
    visit(variable)

    return list(result)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    derivatives = {variable.unique_id : deriv}

    for v in topological_sort(variable):
        
        if v.is_leaf():

            v.accumulate_derivative(derivatives[v.unique_id])
        else:
            for var, d_input in v.chain_rule(derivatives[v.unique_id]):
                derivatives[var.unique_id] = derivatives.get(var.unique_id, 0) + d_input



@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
