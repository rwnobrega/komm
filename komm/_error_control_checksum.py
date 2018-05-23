import numpy as np

__all__ = ['CRC']


class CRC:
    def __init__(self, polynomial):
        self.polynomial = polynomial