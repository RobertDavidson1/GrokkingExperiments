from ..group import Group


# Class for group formed by integers mod n
class Zn(Group):
    def __init__(self, config):
        n = config.task.n

        # Init the parent `Group` class
        super().__init__(index=n, order=n, identity=0, config=config)

    def compose(self, a,b):
        return (a+b) % self.order








