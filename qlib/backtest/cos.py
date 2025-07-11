from .decision import Order
from ast import literal_eval


class CostD:
    def __init__(self, name, expression: str, f: str = None):
        self.name = name
        self.expression = expression
        self.f = f

    def ex(self) -> list[str]:
        return [] if self.f is None else self.f

    def calc(self, trade_val: float, total_trade_val: float) -> float:
        return eval(
            self.expression,
            {"__builtins__": None},
            {
                "$trade_val": trade_val,
                "$total_trade_val": total_trade_val,
            }
        )


class Cost1(CostD):
    def __init__(self, name, cost: float, expression: str = None):
        self.cost = cost
        expression = f"$deal_amount * $deal_price * {self.cost}" if expression is None else expression
        super().__init__(name, expression)


class Cost1(CostD):
    def __init__(self, name, cost: float, expression: str = None):
        self.cost = cost
        expression = f"$deal_amount * $deal_price * {self.cost}" if expression is None else expression
        super().__init__(name, expression)
