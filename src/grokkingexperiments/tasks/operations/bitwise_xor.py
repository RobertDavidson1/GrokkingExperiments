from ..operation import Operation

# Class for bitwise and operation
class BitwiseXor(Operation):
    def compose(self, a, b):
        return a & b
