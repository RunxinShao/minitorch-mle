from typing import Tuple
from .tensor import Tensor
from .tensor_functions import Function, rand
from .autodiff import Context
from . import operators
from .fast_ops import FastOps

# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Tile a tensor into blocks for pooling operations.

    Args:
    ----
        input: Input tensor of shape (batch, channel, height, width)
        kernel: Tuple of (kernel_height, kernel_width)

    Returns:
    -------
        Tuple containing:
        - Tiled tensor of shape (batch, channel, new_height, new_width, kernel_size)
        - New height after tiling
        - New width after tiling

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel

    new_height = height // kh
    new_width = width // kw

    input = input.contiguous()

    input = input.view(batch, channel, new_height, kh, new_width, kw)

    input = input.permute(0, 1, 2, 4, 3, 5).contiguous()

    input = input.view(batch, channel, new_height, new_width, kh * kw)
    return input, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D average pooling on input tensor.

    Args:
    ----
        input: Input tensor of shape (batch, channel, height, width)
        kernel: Tuple of (kernel_height, kernel_width)

    Returns:
    -------
        Output tensor after average pooling

    """
    batch, channel, height, width = input.shape
    t, new_height, new_width = tile(input, kernel)
    t = t.mean(4)
    t = t.view(batch, channel, new_height, new_width)
    return t


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax along a dimension as a 1-hot tensor.

    Args:
    ----
        input: Input tensor
        dim: Dimension to compute argmax along

    Returns:
    -------
        Tensor of the same shape as input, 1.0 where the value equals the max
        along `dim` and 0.0 elsewhere

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    """Function that computes max reduction along a dimension."""

    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for max reduction.

        Args:
        ----
            ctx: Context for saving values
            input: Input tensor
            dim: Dimension to reduce along

        Returns:
        -------
            Max values along specified dimension

        """
        d = int(dim.item())
        ctx.save_for_backward(input, d)
        return max_reduce(input, d)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for max reduction.

        Args:
        ----
            ctx: Context with saved values
            grad_output: Gradient of the output

        Returns:
        -------
            Tuple of gradients for inputs

        """
        input, d = ctx.saved_values
        return grad_output * argmax(input, d), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Compute max reduction along specified dimension.

    Args:
    ----
        input: Input tensor
        dim: Dimension to reduce along

    Returns:
    -------
        Max values along specified dimension

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute softmax along specified dimension.

    Args:
    ----
        input: Input tensor
        dim: Dimension to compute softmax along

    Returns:
    -------
        Softmax probabilities along specified dimension

    """
    # Subtract the max before exp so large inputs do not overflow;
    # softmax is unchanged by a constant shift along `dim`.
    exp_vals = (input - max(input, dim)).exp()
    sums = exp_vals.sum(dim=dim)
    return exp_vals / sums


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute log softmax along specified dimension.

    Args:
    ----
        input: Input tensor
        dim: Dimension to compute log softmax along

    Returns:
    -------
        Log softmax values along specified dimension

    """
    max_vals = max(input, dim)
    shifted = input - max_vals
    log_sums = shifted.exp().sum(dim=dim).log()
    return shifted - log_sums


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D max pooling on input tensor.

    Args:
    ----
        input: Input tensor of shape (batch, channel, height, width)
        kernel: Tuple of (kernel_height, kernel_width)

    Returns:
    -------
        Output tensor after max pooling

    """
    t, new_height, new_width = tile(input, kernel)
    t = max(t, 4)
    return t.view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout to input tensor.

    Args:
    ----
        input: Input tensor
        rate: Dropout rate between 0 and 1
        ignore: If True, return input unchanged

    Returns:
    -------
        Output tensor with dropout applied

    """
    if not ignore:
        rand_tensor = rand(input.shape)
        random_drop = rand_tensor._ensure_tensor(rate) < rand_tensor
        return input * random_drop
    else:
        return input
