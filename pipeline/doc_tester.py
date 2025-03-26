import numpy as np

def tester(arg_1: int=5, arg_2: str='Hello') -> np.ndarray:
    """Tester function for documentation purposes.

    Args:
        arg_1 (int, optional): An integer. Defaults to 5.
        arg_2 (str, optional): A string. Defaults to 'Hello'.
    
    Returns:
        np.ndarray: A numpy array containing arg_1 and arg_2.
    """
    return np.array([arg_1, arg_2])
    