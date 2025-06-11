from qlib.data.ops import ExpressionOps, Rolling, ElemOperator, NpElemOperator
from qlib.data.base import Feature
import pandas as pd
import numpy as np


class Sin(NpElemOperator):

    def __init__(self, feature):
        super().__init__(feature, "sin")


class Cos(NpElemOperator):

    def __init__(self, feature):
        super().__init__(feature, "cos")


class Month(ElemOperator):

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        valid = (series > 0) & (series < 4102444800)
        series = series[valid].astype(np.int64)
        return pd.to_datetime(series, unit="s").dt.month


class Day(ElemOperator):

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        valid = (series > 0) & (series < 4102444800)
        series = series[valid].astype(np.int64)
        return pd.to_datetime(series, unit="s").dt.day


class SAR(Rolling):
    """
    自定义SAR(抛物线转向指标)特征算子,可用于Qlib表达式系统。
    """

    def __init__(self, period, step, max_step):
        """
        period: int, 初始周期
        step: int, 加速因子步长（百分比,如2表示0.02)
        max_step: int, 最大加速因子（百分比,如20表示0.2)
        """
        self.high = Feature("high")
        self.low = Feature("low")

        self.period = period
        self.step = step
        self.max_step = max_step

        super().__init__(self.high, self.period, "sar")

    def __str__(self):
        return f"SAR({self.period},{self.step},{self.max_step})"

    def _load_internal(self, instrument, start_index, end_index, *args):
        high_series = self.high.load(instrument, start_index, end_index, *args)
        low_series = self.low.load(instrument, start_index, end_index, *args)

        assert len(high_series) == len(
            low_series
        ), "High and Low series must have the same length."

        if len(high_series) <= self.period:
            return pd.Series(len(high_prices) * [None])

        high_prices = pd.Series(high_series.values)
        low_prices = pd.Series(low_series.values)

        sars = self._parabolic_sar(
            high_prices,
            low_prices,
            period=self.period,
            step=self.step,
            max_step=self.max_step,
        )

        return pd.Series(sars.values, index=high_series.index)

    def _parabolic_sar(
        self,
        high_prices: pd.Series,
        low_prices: pd.Series,
        is_up: bool = True,
        period: int = 4,
        step: int = 2,
        max_step: int = 20,
    ):
        """
        计算SAR指标
        high_prices, low_prices: pd.Series, 价格序列
        is_up: bool, 初始趋势方向(True为上涨,False为下跌)
        period: int, 初始周期
        step: int, 加速因子步长
        max_step: int, 最大加速因子
        返回: pd.Series, SAR值序列
        """

        # 初始化参数
        step_factor = step / 100
        max_step_factor = max_step / 100
        previous_low = float("inf")  # 上一个周期的最低价
        previous_high = float("-inf")  # 上一个周期的最高价

        # 初始化输出列表
        acceleration_factors = [0] * len(high_prices)
        sar_values = [None] * len(high_prices)

        # 确定初始SAR值
        if is_up:
            sar_values[period - 1] = min(low_prices[: period - 1])

        elif not is_up:
            sar_values[period - 1] = max(high_prices[: period - 1])

        # 计算后续周期的SAR值
        for i in range(period, len(high_prices)):

            # 更新加速因子
            acceleration_factors[i] = acceleration_factors[i - 1] + step_factor
            if acceleration_factors[i] > max_step_factor:
                acceleration_factors[i] = max_step_factor

            # 计算SAR值
            if is_up:
                sar_values[i] = abs(sar_values[i - 1]) + acceleration_factors[i] * (
                    high_prices[i - 1] - abs(sar_values[i - 1])
                )
                previous_high = max(previous_high, high_prices[i])

                # 转势
                if low_prices[i] < abs(sar_values[i]):
                    is_up = False
                    acceleration_factors[i] = 0
                    previous_low = low_prices[i]
                    sar_values[i] = -previous_high

            else:
                sar_values[i] = -(
                    abs(sar_values[i - 1])
                    + acceleration_factors[i]
                    * (low_prices[i - 1] - abs(sar_values[i - 1]))
                )
                previous_low = min(previous_low, low_prices[i])

                # 转势
                if high_prices[i] > abs(sar_values[i]):
                    is_up = True
                    acceleration_factors[i] = 0
                    previous_high = high_prices[i]
                    sar_values[i] = previous_low

        # 返回SAR值
        return pd.Series(sar_values)
