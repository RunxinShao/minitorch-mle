from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to the given values and return a new Scalar."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass of the addition function.

        Args:
        ----
            ctx (Context): The context for the computation.
            a (float): The first operand.
            b (float): The second operand.

        Returns:
        -------
            float: The result of the addition.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass of the addition function.

        Args:
        ----
            ctx (Context): The context for the computation.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, ...]: The derivatives of the inputs.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass of the log function.

        Args:
        ----
            ctx (Context): The context for the computation.
            a (float): The operand.

        Returns:
        -------
            float: The result of the log function.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass of the log function.

        Args:
        ----
            ctx (Context): The context for the computation.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The derivative of the input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


# TODO: Implement for Task 1.2.
class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass of the multiplication function.

        Args:
        ----
            ctx (Context): The context for the computation.
            a (float): The first operand.
            b (float): The second operand.

        Returns:
        -------
            float: The result of the multiplication.

        """
        ctx.save_for_backward(a, b)
        return float(a * b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass of the multiplication function.

        Args:
        ----
            ctx (Context): The context for the computation.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, ...]: The derivatives of the inputs.

        """
        a, b = ctx.saved_values  # Unpack the saved values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass of the inverse function.

        Args:
        ----
            ctx (Context): The context for the computation.
            a (float): The operand.

        Returns:
        -------
            float: The result of the inverse function.

        """
        ctx.save_for_backward(a)
        return 1.0 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass of the inverse function.

        Args:
        ----
            ctx (Context): The context for the computation.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The derivative of the input.

        """
        (a,) = ctx.saved_values
        return -d_output / (a * a)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass of the negation function.

        Args:
        ----
            ctx (Context): The context for the computation.
            a (float): The operand.

        Returns:
        -------
            float: The result of the negation function.

        """
        return -float(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass of the negation function.

        Args:
        ----
            ctx (Context): The context for the computation.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The derivative of the input.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass of the sigmoid function.

        Args:
        ----
            ctx (Context): The context for the computation.
            a (float): The operand.

        Returns:
        -------
            float: The result of the sigmoid function.

        """
        result = 1.0 / (1.0 + operators.exp(-a))
        ctx.save_for_backward(result)  # Save the sigmoid result for backpropagation
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass of the sigmoid function.

        Args:
        ----
            ctx (Context): The context for the computation.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The derivative of the input.

        """
        (result,) = ctx.saved_values  # Unpack the saved value
        return d_output * result * (1.0 - result)


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Compute the forward pass of the ReLU function.

        Args:
        ----
            ctx (Context): The context for the computation.
            x (float): The operand.

        Returns:
        -------
            float: The result of the ReLU function.

        """
        ctx.save_for_backward(x)
        return float(max(0.0, x))  # Ensure the return type is float

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass of the ReLU function.

        Args:
        ----
            ctx (Context): The context for the computation.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The derivative of the input.

        """
        (x,) = ctx.saved_values
        return float(d_output) if x > 0.0 else 0.0  # Use 0.0 instead of 0


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass of the exponential function.

        Args:
        ----
            ctx (Context): The context for the computation.
            a (float): The operand.

        Returns:
        -------
            float: The result of the exponential function.

        """
        ctx.save_for_backward(a)
        return float(operators.exp(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass of the exponential function.

        Args:
        ----
            ctx (Context): The context for the computation.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The derivative of the input.

        """
        (a,) = ctx.saved_values
        return d_output * operators.exp(a)


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass of the less than function.

        Args:
        ----
            ctx (Context): The context for the computation.
            a (float): The first operand.
            b (float): The second operand.

        Returns:
        -------
            float: The result of the less than function.

        """
        return float(a < b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass of the less than function.

        Args:
        ----
            ctx (Context): The context for the computation.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, ...]: The derivatives of the inputs.

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass of the equal function.

        Args:
        ----
            ctx (Context): The context for the computation.
            a (float): The first operand.
            b (float): The second operand.

        Returns:
        -------
            float: The result of the equal function.

        """
        return float(a == b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass of the equal function.

        Args:
        ----
            ctx (Context): The context for the computation.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, ...]: The derivatives of the inputs.

        """
        return 0.0, 0.0
