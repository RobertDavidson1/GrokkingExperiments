from ..operation import Operation

# Class for bitwise and operation
class BitwiseAnd(Operation):
    def compose(self, a, b):
        return a & b
