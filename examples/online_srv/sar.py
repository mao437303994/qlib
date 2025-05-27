from qlib.data.base import ExpressionOps, Feature
from qlib.data.ops import Rolling
import pandas as pd


class SAR(Rolling):

    def __init__(self, period, step, max_step):
        self.period = period
        self.step = step
        self.max_step = max_step

        self.high = Feature("high")
        self.low = Feature("low")

        super().__init__(self.high, self.period, "sar")

    def __str__(self):
        return f"SAR({self.period}, {self.step}, {self.max_step})"

    def _load_internal(self, instrument, start_index, end_index, freq):
        h = self.high.load(instrument, start_index, end_index, freq)
        l = self.low.load(instrument, start_index, end_index, freq)

        assert len(h) == len(
            l), "High and Low prices must have the same length."

        if len(h) < self.period:
            return pd.Series([None] * len(h), index=h.index)

        high_prices = pd.Series(h.values)
        low_prices = pd.Series(l.values)

        sar_values = self._parabolic_sar(
            high_prices, low_prices, period=self.period, step=self.step, max_step=self.max_step)

        return pd.Series(sar_values.values, index=h.index)

    def _parabolic_sar(self, high_prices: pd.Series, low_prices: pd.Series, is_up: bool = True, period: int = 4, step: int = 2, max_step: int = 20):
        """
        计算Parabolic SAR指标。

        参数:
        - high_prices: 列表，表示每个周期的最高价。
        - low_prices: 列表，表示每个周期的最低价。
        - is_up: 布尔值，表示当前趋势是否为上涨，默认为True。
        - period: 整数，表示初始周期长度，默认为4。
        - step: 整数，表示加速因子步长，默认为2。
        - max_step: 整数，表示最大加速因子，默认为20。

        返回:
        - sar_values: 列表，表示每个周期的SAR值。
        """

        # 初始化参数
        step_factor = step / 100
        max_step_factor = max_step / 100
        previous_low = float('inf')  # 上一个周期的最低价
        previous_high = float('-inf')  # 上一个周期的最高价

        # 初始化输出列表
        acceleration_factors = [0] * len(high_prices)
        sar_values = [None] * len(high_prices)

        # 确定初始SAR值
        if is_up:
            sar_values[period-1] = min(low_prices[:period-1])

        elif not is_up:
            sar_values[period-1] = max(high_prices[:period-1])

        # 计算后续周期的SAR值
        for i in range(period, len(high_prices)):

            # 更新加速因子
            acceleration_factors[i] = acceleration_factors[i-1] + step_factor
            if acceleration_factors[i] > max_step_factor:
                acceleration_factors[i] = max_step_factor

            # 计算SAR值
            if is_up:
                sar_values[i] = (abs(sar_values[i-1]) + acceleration_factors[i] *
                                 (high_prices[i-1] - abs(sar_values[i-1])))
                previous_high = max(previous_high, high_prices[i])

                # 转势
                if low_prices[i] < abs(sar_values[i]):
                    is_up = False
                    acceleration_factors[i] = 0
                    previous_low = low_prices[i]
                    sar_values[i] = -previous_high

            else:
                sar_values[i] = -(abs(sar_values[i-1]) + acceleration_factors[i]
                                  * (low_prices[i-1] - abs(sar_values[i-1])))
                previous_low = min(previous_low, low_prices[i])

                # 转势
                if high_prices[i] > abs(sar_values[i]):
                    is_up = True
                    acceleration_factors[i] = 0
                    previous_high = high_prices[i]
                    sar_values[i] = previous_low

        # 返回SAR值
        return pd.Series([sar for sar in sar_values])
