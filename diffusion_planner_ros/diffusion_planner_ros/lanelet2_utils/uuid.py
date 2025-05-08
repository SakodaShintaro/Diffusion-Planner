import hashlib


def uuid(value: str | int, digit: int = 5) -> int:
    """Return the unique identifier in integer.

    If the input uuid is `int`, just returns the same value.
    Otherwise, hash it and generate the specified `digit` length of ID.

    Args:
    ----
        value (str | int): Unique ID in string or int.
        digit (int, optional): Length of Unique ID. Defaults to 5.

    Returns:
    -------
        int: Unique ID in integer.

    Examples:
    --------
        >>> uuid("FOO", 5)
        17715
        >>> uuid(1234)
        1234

    """
    if isinstance(value, str):
        return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16) % 10**digit
    return value
