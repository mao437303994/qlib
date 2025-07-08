# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from abc import abstractmethod
from datetime import time
from enum import IntEnum

# try to fix circular imports when enabling type hints
from typing import TYPE_CHECKING, Any, ClassVar, Generic, List, Optional, Tuple, TypeVar, Union, cast

from qlib.backtest.utils import TradeCalendarManager
from qlib.data.data import Cal
from qlib.log import get_module_logger
from qlib.utils.time import concat_date_time, epsilon_change

if TYPE_CHECKING:
    from qlib.strategy.base import BaseStrategy
    from qlib.backtest.exchange import Exchange

from dataclasses import dataclass

import numpy as np
import pandas as pd

DecisionType = TypeVar("DecisionType")


class OrderDir(IntEnum):
    # Legacy compatibility (保持原有值不变)
    SELL = 0          # 传统卖出/平仓
    BUY = 1           # 传统买入/开仓
    
    # Enhanced closing operations (平仓操作，使用小于10的值)
    SELL_LONG = 2     # 平多仓位
    BUY_SHORT = 3     # 平空仓位
    
    # Enhanced opening operations (开仓操作，使用10+的值)
    BUY_LONG = 10     # 开多仓位
    SELL_SHORT = 11   # 开空仓位


@dataclass
class Order:
    """
    Enhanced Order class for leveraged trading
    
    stock_id : str
    amount : float
    start_time : pd.Timestamp
        closed start time for order trading
    end_time : pd.Timestamp
        closed end time for order trading
    direction : int
        OrderDir.SELL_SHORT/BUY_SHORT for short; OrderDir.BUY_LONG/SELL_LONG for long
        OrderDir.SELL/BUY for legacy compatibility
    leverage : float
        leverage multiplier (default 1.0 for no leverage)
    """

    # 1) time invariant values (required fields)
    # - they are set by users and is time-invariant.
    stock_id: str
    amount: float  # `amount` is a non-negative and adjusted value
    direction: OrderDir
    
    # 2) time variant values (required fields):
    # - Users may want to set these values when using lower level APIs
    # - If users don't, TradeDecisionWO will help users to set them
    # The interval of the order which belongs to (NOTE: this is not the expected order dealing range time)
    start_time: pd.Timestamp
    end_time: pd.Timestamp

    # 3) fields with default values must come after fields without defaults
    leverage: float = 1.0  # 杠杆倍数

    # 4) results
    # - users should not care about these values
    # - they are set by the backtest system after finishing the results.
    # What the value should be about in all kinds of cases
    # - not tradable: the deal_amount == 0 , factor is None
    #    - the stock is suspended and the entire order fails. No cost for this order
    # - dealt or partially dealt: deal_amount >= 0 and factor is not None
    deal_amount: float = 0.0  # `deal_amount` is a non-negative value
    factor: Optional[float] = None

    # TODO:
    # a status field to indicate the dealing result of the order

    # FIXME:
    # for compatible now.
    # Please remove them in the future
    SELL_LONG: ClassVar[OrderDir] = OrderDir.SELL_LONG
    SELL_SHORT: ClassVar[OrderDir] = OrderDir.SELL_SHORT
    BUY_SHORT: ClassVar[OrderDir] = OrderDir.BUY_SHORT
    BUY_LONG: ClassVar[OrderDir] = OrderDir.BUY_LONG
    SELL: ClassVar[OrderDir] = OrderDir.SELL
    BUY: ClassVar[OrderDir] = OrderDir.BUY

    def __post_init__(self) -> None:
        # Support both legacy and new direction types
        valid_directions = {
            OrderDir.SELL, OrderDir.BUY,  # Legacy
            OrderDir.SELL_LONG, OrderDir.SELL_SHORT, 
            OrderDir.BUY_SHORT, OrderDir.BUY_LONG  # New leveraged directions
        }
        if self.direction not in valid_directions:
            raise NotImplementedError(f"direction {self.direction} not supported")
        
        self.deal_amount = 0.0
        self.factor = None
        
        # Validate leverage
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")

    @property
    def amount_delta(self) -> float:
        """
        return the delta of amount considering leverage.
        - Positive value indicates net long position increase
        - Negative value indicates net long position decrease
        """
        return self.amount * self.sign * self.leverage

    @property
    def deal_amount_delta(self) -> float:
        """
        return the delta of deal_amount considering leverage.
        - Positive value indicates net long position increase
        - Negative value indicates net long position decrease
        """
        return self.deal_amount * self.sign * self.leverage

    @property
    def sign(self) -> int:
        """
        return the sign of trading for net position change
        - `+1` indicates net long position increase
        - `-1` indicates net long position decrease
        
        注意：这里的sign是为了与qlib的report.py指标计算兼容
        """
        if self.direction == OrderDir.BUY_LONG:
            return 1   # 开多仓：增加多头净持仓
        elif self.direction == OrderDir.SELL_LONG:
            return -1  # 平多仓：减少多头净持仓
        elif self.direction == OrderDir.SELL_SHORT:
            return -1  # 开空仓：减少多头净持仓(增加空头净持仓)
        elif self.direction == OrderDir.BUY_SHORT:
            return 1   # 平空仓：增加多头净持仓(减少空头净持仓)
        else:
            # Legacy compatibility: BUY=1, SELL=0
            return self.direction * 2 - 1

    @property
    def is_long_direction(self) -> bool:
        """Check if this is a long position related order (包括原有的BUY/SELL)"""
        return self.direction in (
            OrderDir.BUY_LONG, OrderDir.SELL_LONG,  # 明确的多头操作
            OrderDir.BUY, OrderDir.SELL              # 原有操作(默认为做多)
        )
    
    @property
    def is_short_direction(self) -> bool:
        """Check if this is a short position related order"""
        return self.direction in (OrderDir.BUY_SHORT, OrderDir.SELL_SHORT)
    
    @property
    def is_opening_position(self) -> bool:
        """Check if this order opens a new position"""
        return self.direction in (
            OrderDir.BUY_LONG, OrderDir.SELL_SHORT,  # 开多仓/开空仓
            OrderDir.BUY                             # 原有买入(开多仓)
        )
    
    @property
    def is_closing_position(self) -> bool:
        """Check if this order closes an existing position"""
        return self.direction in (
            OrderDir.SELL_LONG, OrderDir.BUY_SHORT,  # 平多仓/平空仓
            OrderDir.SELL                            # 原有卖出(平多仓)
        )

    @property
    def required_margin(self) -> float:
        """
        Calculate required margin for this order
        保证金 = 交易金额 × 保证金率 = 数量 × 价格 × (1/杠杆倍数)
        
        注意：这里只计算基于数量的保证金率，实际保证金需要乘以价格
        """
        margin_rate = 1.0 / self.leverage
        return self.amount * margin_rate
    
    @property 
    def effective_leverage(self) -> float:
        """获取有效杠杆倍数"""
        return self.leverage
    
    @property
    def margin_rate(self) -> float:
        """获取保证金率（根据杠杆倍数自动计算）"""
        return 1.0 / self.leverage

    @staticmethod
    def parse_dir(direction: Union[str, int, np.integer, OrderDir, np.ndarray]) -> Union[OrderDir, np.ndarray]:
        if isinstance(direction, OrderDir):
            return direction
        elif isinstance(direction, (int, float, np.integer, np.floating)):
            # 为了向后兼容，正数仍然映射到BUY，负数映射到SELL
            return Order.BUY if direction > 0 else Order.SELL
        elif isinstance(direction, str):
            dl = direction.lower().strip()
            if dl in ("sell", "sell_long"):
                return OrderDir.SELL_LONG if dl == "sell_long" else OrderDir.SELL
            elif dl in ("buy", "buy_long"):
                return OrderDir.BUY_LONG if dl == "buy_long" else OrderDir.BUY
            elif dl == "sell_short":
                return OrderDir.SELL_SHORT
            elif dl == "buy_short":
                return OrderDir.BUY_SHORT
            else:
                raise NotImplementedError(f"Direction '{direction}' is not supported")
        elif isinstance(direction, np.ndarray):
            direction_array = direction.copy()
            direction_array[direction_array > 0] = Order.BUY
            direction_array[direction_array <= 0] = Order.SELL
            return direction_array
        else:
            raise NotImplementedError(f"This type of input is not supported")

    @property
    def key_by_day(self) -> tuple:
        """A hashable & unique key to identify this order, under the granularity in day."""
        return self.stock_id, self.date, self.direction

    @property
    def key(self) -> tuple:
        """A hashable & unique key to identify this order."""
        return self.stock_id, self.start_time, self.end_time, self.direction

    @property
    def date(self) -> pd.Timestamp:
        """Date of the order."""
        return pd.Timestamp(self.start_time.replace(hour=0, minute=0, second=0))


class OrderHelper:
    """
    Motivation
    - Make generating order easier
        - User may have no knowledge about the adjust-factor information about the system.
        - It involves too much interaction with the exchange when generating orders.
    """

    def __init__(self, exchange: Exchange) -> None:
        self.exchange = exchange

    @staticmethod
    def create(
        code: str,
        amount: float,
        direction: OrderDir,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        leverage: float = 1.0,
    ) -> Order:
        """
        help to create a leveraged order

        Parameters
        ----------
        code : str
            the id of the instrument
        amount : float
            **adjusted trading amount**
        direction : OrderDir
            trading direction (supports both legacy and new leveraged directions)
        start_time : Union[str, pd.Timestamp] (optional)
            The interval of the order which belongs to
        end_time : Union[str, pd.Timestamp] (optional)
            The interval of the order which belongs to
        leverage : float (optional)
            leverage multiplier, default 1.0 (no leverage)

        Returns
        -------
        Order:
            The created leveraged order
        """
        # NOTE: factor is a value belongs to the results section. User don't have to care about it when creating orders
        return Order(
            stock_id=code,
            amount=amount,
            start_time=None if start_time is None else pd.Timestamp(start_time),
            end_time=None if end_time is None else pd.Timestamp(end_time),
            direction=direction,
            leverage=leverage,
        )


class TradeRange:
    @abstractmethod
    def __call__(self, trade_calendar: TradeCalendarManager) -> Tuple[int, int]:
        """
        This method will be call with following way

        The outer strategy give a decision with with `TradeRange`
        The decision will be checked by the inner decision.
        inner decision will pass its trade_calendar as parameter when getting the trading range
        - The framework's step is integer-index based.

        Parameters
        ----------
        trade_calendar : TradeCalendarManager
            the trade_calendar is from inner strategy

        Returns
        -------
        Tuple[int, int]:
            the start index and end index which are tradable

        Raises
        ------
        NotImplementedError:
            Exceptions are raised when no range limitation
        """
        raise NotImplementedError(f"Please implement the `__call__` method")

    @abstractmethod
    def clip_time_range(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Parameters
        ----------
        start_time : pd.Timestamp
        end_time : pd.Timestamp
            Both sides (start_time, end_time) are closed

        Returns
        -------
        Tuple[pd.Timestamp, pd.Timestamp]:
            The tradable time range.
            - It is intersection of [start_time, end_time] and the rule of TradeRange itself
        """
        raise NotImplementedError(f"Please implement the `clip_time_range` method")


class IdxTradeRange(TradeRange):
    def __init__(self, start_idx: int, end_idx: int) -> None:
        self._start_idx = start_idx
        self._end_idx = end_idx

    def __call__(self, trade_calendar: TradeCalendarManager | None = None) -> Tuple[int, int]:
        return self._start_idx, self._end_idx

    def clip_time_range(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
        raise NotImplementedError


class TradeRangeByTime(TradeRange):
    """This is a helper function for make decisions"""

    def __init__(self, start_time: str | time, end_time: str | time) -> None:
        """
        This is a callable class.

        **NOTE**:
        - It is designed for minute-bar for intra-day trading!!!!!
        - Both start_time and end_time are **closed** in the range

        Parameters
        ----------
        start_time : str | time
            e.g. "9:30"
        end_time : str | time
            e.g. "14:30"
        """
        self.start_time = pd.Timestamp(start_time).time() if isinstance(start_time, str) else start_time
        self.end_time = pd.Timestamp(end_time).time() if isinstance(end_time, str) else end_time
        assert self.start_time < self.end_time

    def __call__(self, trade_calendar: TradeCalendarManager) -> Tuple[int, int]:
        if trade_calendar is None:
            raise NotImplementedError("trade_calendar is necessary for getting TradeRangeByTime.")

        start_date = trade_calendar.start_time.date()
        val_start, val_end = concat_date_time(start_date, self.start_time), concat_date_time(start_date, self.end_time)
        return trade_calendar.get_range_idx(val_start, val_end)

    def clip_time_range(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
        start_date = start_time.date()
        val_start, val_end = concat_date_time(start_date, self.start_time), concat_date_time(start_date, self.end_time)
        # NOTE: `end_date` should not be used. Because the `end_date` is for slicing. It may be in the next day
        # Assumption: start_time and end_time is for intra-day trading. So it is OK for only using start_date
        return max(val_start, start_time), min(val_end, end_time)


class BaseTradeDecision(Generic[DecisionType]):
    """
    Trade decisions are made by strategy and executed by executor

    Motivation:
        Here are several typical scenarios for `BaseTradeDecision`

        Case 1:
        1. Outer strategy makes a decision. The decision is not available at the start of current interval
        2. After a period of time, the decision are updated and become available
        3. The inner strategy try to get the decision and start to execute the decision according to `get_range_limit`
        Case 2:
        1. The outer strategy's decision is available at the start of the interval
        2. Same as `case 1.3`
    """

    def __init__(self, strategy: BaseStrategy, trade_range: Union[Tuple[int, int], TradeRange, None] = None) -> None:
        """
        Parameters
        ----------
        strategy : BaseStrategy
            The strategy who make the decision
        trade_range: Union[Tuple[int, int], Callable] (optional)
            The index range for underlying strategy.

            Here are two examples of trade_range for each type

            1) Tuple[int, int]
            start_index and end_index of the underlying strategy(both sides are closed)

            2) TradeRange

        """
        self.strategy = strategy
        self.start_time, self.end_time = strategy.trade_calendar.get_step_time()
        # upper strategy has no knowledge about the sub executor before `_init_sub_trading`
        self.total_step: Optional[int] = None
        if isinstance(trade_range, tuple):
            # for Tuple[int, int]
            trade_range = IdxTradeRange(*trade_range)
        self.trade_range: Optional[TradeRange] = trade_range

    def get_decision(self) -> List[DecisionType]:
        """
        get the **concrete decision**  (e.g. execution orders)
        This will be called by the inner strategy

        Returns
        -------
        List[DecisionType:
            The decision result. Typically it is some orders
            Example:
                []:
                    Decision not available
                [concrete_decision]:
                    available
        """
        raise NotImplementedError(f"This type of input is not supported")

    def update(self, trade_calendar: TradeCalendarManager) -> Optional[BaseTradeDecision]:
        """
        Be called at the **start** of each step.

        This function is design for following purpose
        1) Leave a hook for the strategy who make `self` decision to update the decision itself
        2) Update some information from the inner executor calendar

        Parameters
        ----------
        trade_calendar : TradeCalendarManager
            The calendar of the **inner strategy**!!!!!

        Returns
        -------
        BaseTradeDecision:
            New update, use new decision. If no updates, return None (use previous decision (or unavailable))
        """
        # purpose 1)
        self.total_step = trade_calendar.get_trade_len()

        # purpose 2)
        return self.strategy.update_trade_decision(self, trade_calendar)

    def _get_range_limit(self, **kwargs: Any) -> Tuple[int, int]:
        if self.trade_range is not None:
            return self.trade_range(trade_calendar=cast(TradeCalendarManager, kwargs.get("inner_calendar")))
        else:
            raise NotImplementedError("The decision didn't provide an index range")

    def get_range_limit(self, **kwargs: Any) -> Tuple[int, int]:
        """
        return the expected step range for limiting the decision execution time
        Both left and right are **closed**

        if no available trade_range, `default_value` will be returned

        It is only used in `NestedExecutor`
        - The outmost strategy will not follow any range limit (but it may give range_limit)
        - The inner most strategy's range_limit will be useless due to atomic executors don't have such
          features.

        **NOTE**:
        1) This function must be called after `self.update` in following cases(ensured by NestedExecutor):
        - user relies on the auto-clip feature of `self.update`

        2) This function will be called after _init_sub_trading in NestedExecutor.

        Parameters
        ----------
        **kwargs:
            {
                "default_value": <default_value>, # using dict is for distinguish no value provided or None provided
                "inner_calendar": <trade calendar of inner strategy>
                # because the range limit  will control the step range of inner strategy, inner calendar will be a
                # important parameter when trade_range is callable
            }

        Returns
        -------
        Tuple[int, int]:

        Raises
        ------
        NotImplementedError:
            If the following criteria meet
            1) the decision can't provide a unified start and end
            2) default_value is not provided
        """
        try:
            _start_idx, _end_idx = self._get_range_limit(**kwargs)
        except NotImplementedError as e:
            if "default_value" in kwargs:
                return kwargs["default_value"]
            else:
                # Default to get full index
                raise NotImplementedError(f"The decision didn't provide an index range") from e

        # clip index
        if getattr(self, "total_step", None) is not None:
            # if `self.update` is called.
            # Then the _start_idx, _end_idx should be clipped
            assert self.total_step is not None
            if _start_idx < 0 or _end_idx >= self.total_step:
                logger = get_module_logger("decision")
                logger.warning(
                    f"[{_start_idx},{_end_idx}] go beyond the total_step({self.total_step}), it will be clipped.",
                )
                _start_idx, _end_idx = max(0, _start_idx), min(self.total_step - 1, _end_idx)
        return _start_idx, _end_idx

    def get_data_cal_range_limit(self, rtype: str = "full", raise_error: bool = False) -> Tuple[int, int]:
        """
        get the range limit based on data calendar

        NOTE: it is **total** range limit instead of a single step

        The following assumptions are made
        1) The frequency of the exchange in common_infra is the same as the data calendar
        2) Users want the index mod by **day** (i.e. 240 min)

        Parameters
        ----------
        rtype: str
            - "full": return the full limitation of the decision in the day
            - "step": return the limitation of current step

        raise_error: bool
            True: raise error if no trade_range is set
            False: return full trade calendar.

            It is useful in following cases
            - users want to follow the order specific trading time range when decision level trade range is not
              available. Raising NotImplementedError to indicates that range limit is not available

        Returns
        -------
        Tuple[int, int]:
            the range limit in data calendar

        Raises
        ------
        NotImplementedError:
            If the following criteria meet
            1) the decision can't provide a unified start and end
            2) raise_error is True
        """
        # potential performance issue
        day_start = pd.Timestamp(self.start_time.date())
        day_end = epsilon_change(day_start + pd.Timedelta(days=1))
        freq = self.strategy.trade_exchange.freq
        _, _, day_start_idx, day_end_idx = Cal.locate_index(day_start, day_end, freq=freq)
        if self.trade_range is None:
            if raise_error:
                raise NotImplementedError(f"There is no trade_range in this case")
            else:
                return 0, day_end_idx - day_start_idx
        else:
            if rtype == "full":
                val_start, val_end = self.trade_range.clip_time_range(day_start, day_end)
            elif rtype == "step":
                val_start, val_end = self.trade_range.clip_time_range(self.start_time, self.end_time)
            else:
                raise ValueError(f"This type of input {rtype} is not supported")
            _, _, start_idx, end_index = Cal.locate_index(val_start, val_end, freq=freq)
            return start_idx - day_start_idx, end_index - day_start_idx

    def empty(self) -> bool:
        for obj in self.get_decision():
            if isinstance(obj, Order):
                # Zero amount order will be treated as empty
                if obj.amount > 1e-6:
                    return False
            else:
                return True
        return True

    def mod_inner_decision(self, inner_trade_decision: BaseTradeDecision) -> None:
        """
        This method will be called on the inner_trade_decision after it is generated.
        `inner_trade_decision` will be changed **inplace**.

        Motivation of the `mod_inner_decision`
        - Leave a hook for outer decision to affect the decision generated by the inner strategy
            - e.g. the outmost strategy generate a time range for trading. But the upper layer can only affect the
              nearest layer in the original design.  With `mod_inner_decision`, the decision can passed through multiple
              layers

        Parameters
        ----------
        inner_trade_decision : BaseTradeDecision
        """
        # base class provide a default behaviour to modify inner_trade_decision
        # trade_range should be propagated when inner trade_range is not set
        if inner_trade_decision.trade_range is None:
            inner_trade_decision.trade_range = self.trade_range


class EmptyTradeDecision(BaseTradeDecision[object]):
    def get_decision(self) -> List[object]:
        return []

    def empty(self) -> bool:
        return True


class TradeDecisionWO(BaseTradeDecision[Order]):
    """
    Trade Decision (W)ith (O)rder.
    Besides, the time_range is also included.
    """

    def __init__(
        self,
        order_list: List[Order],
        strategy: BaseStrategy,
        trade_range: Union[Tuple[int, int], TradeRange, None] = None,
    ) -> None:
        super().__init__(strategy, trade_range=trade_range)
        self.order_list = cast(List[Order], order_list)
        start, end = strategy.trade_calendar.get_step_time()
        for o in order_list:
            assert isinstance(o, Order)
            if o.start_time is None:
                o.start_time = start
            if o.end_time is None:
                o.end_time = end

    def get_decision(self) -> List[Order]:
        return self.order_list

    def __repr__(self) -> str:
        return (
            f"class: {self.__class__.__name__}; "
            f"strategy: {self.strategy}; "
            f"trade_range: {self.trade_range}; "
            f"order_list[{len(self.order_list)}]"
        )


class TradeDecisionWithDetails(TradeDecisionWO):
    """
    Decision with detail information.
    Detail information is used to generate execution reports.
    """

    def __init__(
        self,
        order_list: List[Order],
        strategy: BaseStrategy,
        trade_range: Optional[Tuple[int, int]] = None,
        details: Optional[Any] = None,
    ) -> None:
        super().__init__(order_list, strategy, trade_range)

        self.details = details
