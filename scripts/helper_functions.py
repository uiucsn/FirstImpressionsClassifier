import numpy as np
import os

def remove_prefix(text, prefix):
    """A function to remove a prefix from a string. This is implemented as a built-in
    function in Python3, but adding it here to be safe.

    Parameters
    ----------
    text : str
        The full string.
    prefix : str
        The prefix to remove

    Returns
    -------
    str
        Text without the prefix.

    """
    return text[text.startswith(prefix) and len(prefix):]

def merge_dicts(x):
    """A quick function to combine dictionaries.

    Parameters
    ----------
    x : list or array-like
        A list of dictionaries to merge.

    Returns
    -------
    dict
        The merged dictionary.

    """
    z = x[0].copy()   # start with keys and values of x
    for y in x[1:]:
        z.update(y)    # modifies z with keys and values of y
    return z
