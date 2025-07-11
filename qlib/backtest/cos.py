class CostD:
    def __init__(self, name, expression: str = None):
        self.name = name
        self.expression = expression

    def ex(self) -> list[str]:
        return set()

    def calc(self):
        raise NotImplementedError("calc method should be implemented in subclasses")


class Cost(CostD):
    def __init__(self, name, expression: str = None, cost: float = 0.0095):
        super().__init__(name, expression)
        self.cost = cost

    def calc(self):

        return self.cost