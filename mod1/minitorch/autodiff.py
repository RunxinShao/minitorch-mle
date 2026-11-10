from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol, List, Set


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    """Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f: The function to differentiate.
        *vals: The values at which to evaluate the function.
        arg: The index of the argument to differentiate.
        epsilon: A small constant for the finite difference approximation.

    Returns:
    -------
        An approximation of the derivative.

    """
    # TODO: Implement for Task 1.1.
    vals_list = list(vals)

    # 创建向前和向后的扰动参数列表
    vals_forward = vals_list[:]
    vals_backward = vals_list[:]

    # 只修改指定的参数
    vals_forward[arg] += epsilon
    vals_backward[arg] -= epsilon

    # 在扰动后的参数下计算函数值
    forward = f(*vals_forward)
    backward = f(*vals_backward)

    # 计算中央差分近似值
    return (forward - backward) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of the variable with respect to the output."""
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """Checks if the variable is a leaf node in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """Checks if the variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parents of the variable in the computation graph."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute the derivative of the variable with respect to the output."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    ordered: List[Variable] = []
    visited: Set[int] = set()

    def dfs(var: Variable) -> None:
        if var.unique_id in visited or var.is_constant():
            return
        visited.add(var.unique_id)
        for parent in var.parents:
            dfs(parent)
        ordered.append(var)

    dfs(variable)
    return reversed(ordered)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable.
        deriv:Its derivative that we want to propagate backward to the leaves.

    """
    sorted_variables = topological_sort(variable)
    derivatives = {variable.unique_id: deriv}

    for var in sorted_variables:
        if var.is_leaf():
            var.accumulate_derivative(derivatives[var.unique_id])
        else:
            for parent, grad in var.chain_rule(derivatives[var.unique_id]):
                if parent.unique_id not in derivatives:
                    derivatives[parent.unique_id] = grad
                else:
                    derivatives[parent.unique_id] += grad


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors for backpropagation."""
        return self.saved_values
