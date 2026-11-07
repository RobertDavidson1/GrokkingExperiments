from . base import Group


# Class for group formed by integers mod n
class Zn(Group):
    def __init__(self, n):
        super().__init__(index=n, order=n, identity=0)

    def compose(self, a,b):
        return (a+b) % self.order







