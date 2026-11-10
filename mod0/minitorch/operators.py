"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b


def id(a: float) -> float:
    """Returns the identity of a number."""
    return a


def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b


def neg(a: float) -> float:
    """Negates a number."""
    return -a


def lt(a: float, b: float) -> bool:
    """Compares two numbers."""
    return a < b


def eq(a: float, b: float) -> bool:
    """Compares two numbers"""
    return a == b


def max(a: float, b: float) -> float:
    """Returns the maximum of two numbers."""
    return a if a > b else b


def is_close(a: float, b: float) -> bool:
    """Compares two numbers."""
    return abs(a - b) < 1e-2


def sigmoid(a: float) -> float:
    """Returns the sigmoid of a number."""
    return 1.0 / (1.0 + math.exp(-a)) if a >= 0 else math.exp(a) / (1.0 + math.exp(a))


def relu(a: float) -> float:
    """Returns the ReLU of a number."""
    return a if a > 0 else 0


def log(a: float) -> float:
    """Returns the logarithm of a number."""
    return math.log(a)


def exp(a: float) -> float:
    """Returns the exponential of a number."""
    return math.exp(a)


def log_back(a: float, dout: float) -> float:
    """Returns the derivative of the logarithm of a number."""
    return dout / a


def inv(a: float) -> float:
    """Returns the inverse of a number."""
    return 1.0 / a


def inv_back(a: float, dout: float) -> float:
    """Returns the derivative of the inverse of a number."""
    return -dout / a**2


def relu_back(a: float, dout: float) -> float:
    """Returns the derivative of the ReLU of a number."""
    return dout * (a > 0)


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
# Task 0.3: Higher-order functions


def map(fn: Callable[[float], float], lst: Iterable[float]) -> Iterable[float]:
    """Applies a function to each element of an iterable."""
    return [fn(x) for x in lst]


def zipWith(
    fn: Callable[[float, float], float], lst1: Iterable[float], lst2: Iterable[float]
) -> Iterable[float]:
    """Combines two iterables element-wise using a given function."""
    return [fn(x, y) for x, y in zip(lst1, lst2)]


def reduce(
    fn: Callable[[float, float], float], lst: Iterable[float], start: float
) -> float:
    """Reduces an iterable to a single value using a given function."""
    result = start
    for x in lst:
        result = fn(result, x)
    return result


# Use the higher-order functions to implement the following


def negList(lst: Iterable[float]) -> Iterable[float]:
    """Negates all elements in a list."""
    return map(neg, lst)


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Adds corresponding elements from two lists."""
    return zipWith(add, lst1, lst2)


def sum(lst: Iterable[float]) -> float:
    """Sums all elements in a list."""
    return reduce(add, lst, 0)


def prod(lst: Iterable[float]) -> float:
    """Calculates the product of all elements in a list."""
    return reduce(mul, lst, 1)
