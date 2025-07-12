class CostD:

    @property
    def field(self):
        return self._field

    def __init__(self, name="cost", field: str | float = None):
        self._name = name
        self._field = field

    def ex(self) -> list[str]:
        if self._field is None:
            return []
        return [self._field]


class Cost(CostD):

    def __init__(self, name, cost_ratio: float):
        super().__init__(name, f"CONST('{name}',{cost_ratio})")
