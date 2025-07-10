import numpy as np


def tester(arg_1: int = 5, arg_2: str = "Hello") -> np.ndarray:
    """Tester function for documentation purposes.

    Args:
        arg_1 (int, optional): An integer. Defaults to 5.
        arg_2 (str, optional): A string. Defaults to 'Hello'.

    Returns:
        np.ndarray: A numpy array containing arg_1 and arg_2.
    """
    return np.array([arg_1, arg_2])


def calculate_discounted_price(price: float, discount_rate: float) -> float:
    """
    TO CHECK LINTER: Calculate the discounted price given the original price and discount rate.

    Args:
      price: The original price of the item.
      discount_rate: The discount rate as a fraction (e.g., 0.2 for 20% discount).

    Returns:
      The discounted price as a float.

    Raises:
      ValueError: If the price is negative.
    """
    if price < 0:
        raise ValueError("Price cannot be negative")
    discounted_price: float = price - (price * discount_rate)
    return discounted_price
