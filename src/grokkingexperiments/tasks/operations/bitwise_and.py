from ..task import Task

# Class for bitwise and operation
class BitwiseAnd(Task):
    def compose(self, a, b):
        return a & b
