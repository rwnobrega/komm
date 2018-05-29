def tag(**tags):
    """
    See PEP 232
    """
    def a(function):
        for key, value in tags.items():
            setattr(function, key, value)
        return function
    return a
