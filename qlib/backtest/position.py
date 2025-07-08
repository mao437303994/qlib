# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from ..data.data import D
from .decision import Order


class BasePosition:
    """
    The Position wants to maintain the position like a dictionary
    Please refer to the `Position` class for the position
    """

    def __init__(self, *args: Any, cash: float = 0.0, **kwargs: Any) -> None:
        self._settle_type = self.ST_NO
        self.position: dict = {}

    def fill_stock_value(self, start_time: Union[str, pd.Timestamp], freq: str, last_days: int = 30) -> None:
        pass

    def skip_update(self) -> bool:
        """
        Should we skip updating operation for this position
        For example, updating is meaningless for InfPosition

        Returns
        -------
        bool:
            should we skip the updating operator
        """
        return False

    def check_stock(self, stock_id: str) -> bool:
        """
        check if is the stock in the position

        Parameters
        ----------
        stock_id : str
            the id of the stock

        Returns
        -------
        bool:
            if is the stock in the position
        """
        raise NotImplementedError(f"Please implement the `check_stock` method")

    def update_order(self, order: Order, trade_val: float, cost: float, trade_price: float) -> None:
        """
        Parameters
        ----------
        order : Order
            the order to update the position
        trade_val : float
            the trade value(money) of dealing results
        cost : float
            the trade cost of the dealing results
        trade_price : float
            the trade price of the dealing results
        """
        raise NotImplementedError(f"Please implement the `update_order` method")

    def update_stock_price(self, stock_id: str, price: float) -> None:
        """
        Updating the latest price of the order
        The useful when clearing balance at each bar end

        Parameters
        ----------
        stock_id :
            the id of the stock
        price : float
            the price to be updated
        """
        raise NotImplementedError(f"Please implement the `update stock price` method")

    def calculate_stock_value(self) -> float:
        """
        calculate the value of the all assets except cash in the position

        Returns
        -------
        float:
            the value(money) of all the stock
        """
        raise NotImplementedError(f"Please implement the `calculate_stock_value` method")

    def calculate_value(self) -> float:
        raise NotImplementedError(f"Please implement the `calculate_value` method")

    def get_stock_list(self) -> List[str]:
        """
        Get the list of stocks in the position.
        """
        raise NotImplementedError(f"Please implement the `get_stock_list` method")

    def get_stock_price(self, code: str) -> float:
        """
        get the latest price of the stock

        Parameters
        ----------
        code :
            the code of the stock
        """
        raise NotImplementedError(f"Please implement the `get_stock_price` method")

    def get_stock_amount(self, code: str) -> float:
        """
        get the amount of the stock

        Parameters
        ----------
        code :
            the code of the stock

        Returns
        -------
        float:
            the amount of the stock
        """
        raise NotImplementedError(f"Please implement the `get_stock_amount` method")

    def get_cash(self, include_settle: bool = False) -> float:
        """
        Parameters
        ----------
        include_settle:
            will the unsettled(delayed) cash included
            Default: not include those unavailable cash

        Returns
        -------
        float:
            the available(tradable) cash in position
        """
        raise NotImplementedError(f"Please implement the `get_cash` method")

    def get_stock_amount_dict(self) -> dict:
        """
        generate stock amount dict {stock_id : amount of stock}

        Returns
        -------
        Dict:
            {stock_id : amount of stock}
        """
        raise NotImplementedError(f"Please implement the `get_stock_amount_dict` method")

    def get_stock_weight_dict(self, only_stock: bool = False) -> dict:
        """
        generate stock weight dict {stock_id : value weight of stock in the position}
        it is meaningful in the beginning or the end of each trade step
        - During execution of each trading step, the weight may be not consistent with the portfolio value

        Parameters
        ----------
        only_stock : bool
            If only_stock=True, the weight of each stock in total stock will be returned
            If only_stock=False, the weight of each stock in total assets(stock + cash) will be returned

        Returns
        -------
        Dict:
            {stock_id : value weight of stock in the position}
        """
        raise NotImplementedError(f"Please implement the `get_stock_weight_dict` method")

    def add_count_all(self, bar: str) -> None:
        """
        Will be called at the end of each bar on each level

        Parameters
        ----------
        bar :
            The level to be updated
        """
        raise NotImplementedError(f"Please implement the `add_count_all` method")

    def update_weight_all(self) -> None:
        """
        Updating the position weight;

        # TODO: this function is a little weird. The weight data in the position is in a wrong state after dealing order
        # and before updating weight.
        """
        raise NotImplementedError(f"Please implement the `add_count_all` method")

    ST_CASH = "cash"
    ST_NO = "None"  # String is more typehint friendly than None

    def settle_start(self, settle_type: str) -> None:
        """
        settlement start
        It will act like start and commit a transaction

        Parameters
        ----------
        settle_type : str
            Should we make delay the settlement in each execution (each execution will make the executor a step forward)
            - "cash": make the cash settlement delayed.
                - The cash you get can't be used in current step (e.g. you can't sell a stock to get cash to buy another
                        stock)
            - None: not settlement mechanism
            - TODO: other assets will be supported in the future.
        """
        raise NotImplementedError(f"Please implement the `settle_conf` method")

    def settle_commit(self) -> None:
        """
        settlement commit
        """
        raise NotImplementedError(f"Please implement the `settle_commit` method")

    def __str__(self) -> str:
        return self.__dict__.__str__()

    def __repr__(self) -> str:
        return self.__dict__.__repr__()


class Position(BasePosition):
    """Position

    current state of position
    a typical example is :{
      <instrument_id>: {
        'count': <how many days the security has been hold>,
        'amount': <the amount of the security>,
        'price': <the close price of security in the last trading day>,
        'weight': <the security weight of total position value>,
      },
    }
    """

    def __init__(self, cash: float = 0, position_dict: Dict[str, Union[Dict[str, float], float]] = {}) -> None:
        """Init position by cash and position_dict.

        Parameters
        ----------
        cash : float, optional
            initial cash in account, by default 0
        position_dict : Dict[
                            stock_id,
                            Union[
                                int,  # it is equal to {"amount": int}
                                {"amount": int, "price"(optional): float},
                            ]
                        ]
            initial stocks with parameters amount and price,
            if there is no price key in the dict of stocks, it will be filled by _fill_stock_value.
            by default {}.
        """
        super().__init__()

        # NOTE: The position dict must be copied!!!
        # Otherwise the initial value
        self.init_cash = cash
        self.position = position_dict.copy()
        for stock, value in self.position.items():
            if isinstance(value, int):
                self.position[stock] = {"amount": value}
        self.position["cash"] = cash

        # If the stock price information is missing, the account value will not be calculated temporarily
        try:
            self.position["now_account_value"] = self.calculate_value()
        except KeyError:
            pass

    def fill_stock_value(self, start_time: Union[str, pd.Timestamp], freq: str, last_days: int = 30) -> None:
        """fill the stock value by the close price of latest last_days from qlib.

        Parameters
        ----------
        start_time :
            the start time of backtest.
        freq : str
            Frequency
        last_days : int, optional
            the days to get the latest close price, by default 30.
        """
        stock_list = []
        for stock, value in self.position.items():
            if not isinstance(value, dict):
                continue
            if value.get("price", None) is None:
                stock_list.append(stock)

        if len(stock_list) == 0:
            return

        start_time = pd.Timestamp(start_time)
        # note that start time is 2020-01-01 00:00:00 if raw start time is "2020-01-01"
        price_end_time = start_time
        price_start_time = start_time - timedelta(days=last_days)
        price_df = D.features(
            stock_list,
            ["$close"],
            price_start_time,
            price_end_time,
            freq=freq,
            disk_cache=True,
        ).dropna()
        price_dict = price_df.groupby(["instrument"], group_keys=False).tail(1)["$close"].to_dict()

        if len(price_dict) < len(stock_list):
            lack_stock = set(stock_list) - set(price_dict)
            raise ValueError(f"{lack_stock} doesn't have close price in qlib in the latest {last_days} days")

        for stock in stock_list:
            self.position[stock]["price"] = price_dict[stock]
        self.position["now_account_value"] = self.calculate_value()

    def _init_stock(self, stock_id: str, amount: float, price: float | None = None) -> None:
        """
        initialization the stock in current position

        Parameters
        ----------
        stock_id :
            the id of the stock
        amount : float
            the amount of the stock
        price :
             the price when buying the init stock
        """
        self.position[stock_id] = {}
        self.position[stock_id]["amount"] = amount
        self.position[stock_id]["price"] = price
        self.position[stock_id]["weight"] = 0  # update the weight in the end of the trade date

    def _buy_stock(self, stock_id: str, trade_val: float, cost: float, trade_price: float) -> None:
        trade_amount = trade_val / trade_price
        if stock_id not in self.position:
            self._init_stock(stock_id=stock_id, amount=trade_amount, price=trade_price)
        else:
            # exist, add amount
            self.position[stock_id]["amount"] += trade_amount

        self.position["cash"] -= trade_val + cost

    def _sell_stock(self, stock_id: str, trade_val: float, cost: float, trade_price: float) -> None:
        trade_amount = trade_val / trade_price
        if stock_id not in self.position:
            raise KeyError("{} not in current position".format(stock_id))
        else:
            if np.isclose(self.position[stock_id]["amount"], trade_amount):
                # Selling all the stocks
                # we use np.isclose instead of abs(<the final amount>) <= 1e-5  because `np.isclose` consider both
                # relative amount and absolute amount
                # Using abs(<the final amount>) <= 1e-5 will result in error when the amount is large
                self._del_stock(stock_id)
            else:
                # decrease the amount of stock
                self.position[stock_id]["amount"] -= trade_amount
                # check if to delete
                if self.position[stock_id]["amount"] < -1e-5:
                    raise ValueError(
                        "only have {} {}, require {}".format(
                            self.position[stock_id]["amount"] + trade_amount,
                            stock_id,
                            trade_amount,
                        ),
                    )

        new_cash = trade_val - cost
        if self._settle_type == self.ST_CASH:
            self.position["cash_delay"] += new_cash
        elif self._settle_type == self.ST_NO:
            self.position["cash"] += new_cash
        else:
            raise NotImplementedError(f"This type of input is not supported")

    def _del_stock(self, stock_id: str) -> None:
        del self.position[stock_id]

    def check_stock(self, stock_id: str) -> bool:
        return stock_id in self.position

    def update_order(self, order: Order, trade_val: float, cost: float, trade_price: float) -> None:
        # handle order, order is a order class, defined in exchange.py
        if order.direction in [Order.BUY, Order.BUY_LONG]:
            # BUY or BUY_LONG (开多仓)
            self._buy_stock(order.stock_id, trade_val, cost, trade_price)
        elif order.direction in [Order.SELL, Order.SELL_LONG]:
            # SELL or SELL_LONG (平多仓)
            self._sell_stock(order.stock_id, trade_val, cost, trade_price)
        elif order.direction == Order.SELL_SHORT:
            # SELL_SHORT (开空仓) - 相当于"卖出"股票建立空头仓位
            self._sell_stock_allow_short(order.stock_id, trade_val, cost, trade_price)
        elif order.direction == Order.BUY_SHORT:
            # BUY_SHORT (平空仓) - 相当于"买入"股票平掉空头仓位
            self._buy_stock_cover_short(order.stock_id, trade_val, cost, trade_price)
        else:
            raise NotImplementedError("do not support order direction {}".format(order.direction))

    def update_stock_price(self, stock_id: str, price: float) -> None:
        self.position[stock_id]["price"] = price

    def update_stock_count(self, stock_id: str, bar: str, count: float) -> None:  # TODO: check type of `bar`
        self.position[stock_id][f"count_{bar}"] = count

    def update_stock_weight(self, stock_id: str, weight: float) -> None:
        self.position[stock_id]["weight"] = weight

    def calculate_stock_value(self) -> float:
        stock_list = self.get_stock_list()
        value = 0
        for stock_id in stock_list:
            value += self.position[stock_id]["amount"] * self.position[stock_id]["price"]
        return value

    def calculate_value(self) -> float:
        value = self.calculate_stock_value()
        value += self.position["cash"] + self.position.get("cash_delay", 0.0)
        return value

    def get_stock_list(self) -> List[str]:
        stock_list = list(set(self.position.keys()) - {"cash", "now_account_value", "cash_delay"})
        return stock_list

    def get_stock_price(self, code: str) -> float:
        return self.position[code]["price"]

    def get_stock_amount(self, code: str) -> float:
        return self.position[code]["amount"] if code in self.position else 0

    def get_stock_count(self, code: str, bar: str) -> float:
        """the days the account has been hold, it may be used in some special strategies"""
        if f"count_{bar}" in self.position[code]:
            return self.position[code][f"count_{bar}"]
        else:
            return 0

    def get_stock_weight(self, code: str) -> float:
        return self.position[code]["weight"]

    def get_cash(self, include_settle: bool = False) -> float:
        cash = self.position["cash"]
        if include_settle:
            cash += self.position.get("cash_delay", 0.0)
        return cash

    def get_stock_amount_dict(self) -> dict:
        """generate stock amount dict {stock_id : amount of stock}"""
        d = {}
        stock_list = self.get_stock_list()
        for stock_code in stock_list:
            d[stock_code] = self.get_stock_amount(code=stock_code)
        return d

    def get_stock_weight_dict(self, only_stock: bool = False) -> dict:
        """get_stock_weight_dict
        generate stock weight dict {stock_id : value weight of stock in the position}
        it is meaningful in the beginning or the end of each trade date

        :param only_stock: If only_stock=True, the weight of each stock in total stock will be returned
                           If only_stock=False, the weight of each stock in total assets(stock + cash) will be returned
        """
        if only_stock:
            position_value = self.calculate_stock_value()
        else:
            position_value = self.calculate_value()
        d = {}
        stock_list = self.get_stock_list()
        for stock_code in stock_list:
            d[stock_code] = self.position[stock_code]["amount"] * self.position[stock_code]["price"] / position_value
        return d

    def add_count_all(self, bar: str) -> None:
        stock_list = self.get_stock_list()
        for code in stock_list:
            if f"count_{bar}" in self.position[code]:
                self.position[code][f"count_{bar}"] += 1
            else:
                self.position[code][f"count_{bar}"] = 1

    def update_weight_all(self) -> None:
        weight_dict = self.get_stock_weight_dict()
        for stock_code, weight in weight_dict.items():
            self.update_stock_weight(stock_code, weight)

    def settle_start(self, settle_type: str) -> None:
        assert self._settle_type == self.ST_NO, "Currently, settlement can't be nested!!!!!"
        self._settle_type = settle_type
        if settle_type == self.ST_CASH:
            self.position["cash_delay"] = 0.0

    def settle_commit(self) -> None:
        if self._settle_type != self.ST_NO:
            if self._settle_type == self.ST_CASH:
                self.position["cash"] += self.position["cash_delay"]
                del self.position["cash_delay"]
            else:
                raise NotImplementedError(f"This type of input is not supported")
            self._settle_type = self.ST_NO

    def _sell_stock_allow_short(self, stock_id: str, trade_val: float, cost: float, trade_price: float) -> None:
        """
        开空仓：允许将持仓变为负值
        """
        trade_amount = trade_val / trade_price
        if stock_id not in self.position:
            # 初始化空头仓位 (负持仓)
            self._init_stock(stock_id=stock_id, amount=-trade_amount, price=trade_price)
        else:
            # 减少持仓数量 (可能变为负值)
            self.position[stock_id]["amount"] -= trade_amount

        # 卖空获得现金
        new_cash = trade_val - cost
        if self._settle_type == self.ST_CASH:
            self.position["cash_delay"] += new_cash
        elif self._settle_type == self.ST_NO:
            self.position["cash"] += new_cash
        else:
            raise NotImplementedError(f"This type of input is not supported")

    def _buy_stock_cover_short(self, stock_id: str, trade_val: float, cost: float, trade_price: float) -> None:
        """
        平空仓：买入股票平掉空头仓位
        """
        trade_amount = trade_val / trade_price
        if stock_id not in self.position:
            raise KeyError("{} not in current position".format(stock_id))
        else:
            # 增加持仓数量 (从负值向零或正值移动)
            self.position[stock_id]["amount"] += trade_amount
            
            # 检查是否完全平仓
            if np.isclose(self.position[stock_id]["amount"], 0, atol=1e-5):
                self._del_stock(stock_id)

        # 买入需要支付现金
        self.position["cash"] -= trade_val + cost


class LeveragedPosition(BasePosition):
    """
    杠杆交易仓位类，支持期货、加密货币等多种资产的做多做空和杠杆交易
    
    特性：
    - 支持做多/做空
    - 支持杠杆交易（资金放大）
    - 分别跟踪多头和空头仓位
    - 实时计算未实现盈亏
    - 适用于期货、加密货币、差价合约等
    """
    
    def __init__(self, cash: float = 0, position_dict: Dict[str, Union[Dict[str, float], float]] = None, 
                 default_leverage: float = 1.0) -> None:
        """
        初始化杠杆仓位
        
        Parameters
        ----------
        cash : float
            初始现金
        position_dict : Dict
            初始仓位字典
        default_leverage : float
            默认杠杆倍数
        """
        super().__init__()
        self.init_cash = cash
        self.default_leverage = default_leverage
        self.position = {}
        
        # 初始化现金和统计
        self.position['cash'] = cash
        self.position['total_margin'] = 0.0
        self.position['total_equity'] = cash
        
        # 初始化仓位
        if position_dict:
            for stock, value in position_dict.items():
                if isinstance(value, dict):
                    # 详细仓位信息
                    self._init_leveraged_stock(
                        stock_id=stock,
                        long_amount=value.get('long_amount', 0.0),
                        short_amount=value.get('short_amount', 0.0),
                        leverage=value.get('leverage', default_leverage)
                    )
                else:
                    # 简单数量格式
                    amount = float(value)
                    if amount > 0:
                        self._init_leveraged_stock(stock_id=stock, long_amount=amount)
                    elif amount < 0:
                        self._init_leveraged_stock(stock_id=stock, short_amount=abs(amount))

    def _init_leveraged_stock(self, stock_id: str, long_amount: float = 0.0, short_amount: float = 0.0, 
                            leverage: float = None) -> None:
        """初始化杠杆股票仓位"""
        if leverage is None:
            leverage = self.default_leverage
            
        self.position[stock_id] = {
            'long_amount': long_amount,
            'short_amount': short_amount,
            'net_amount': long_amount - short_amount,
            'leverage': leverage,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'total_cost': 0.0,
            'margin_used': 0.0,
            'weight': 0.0,
            'count': 0,
            'price': 0.0,
        }

    def skip_update(self) -> bool:
        return False

    def check_stock(self, stock_id: str) -> bool:
        return stock_id in self.position

    def update_order(self, order: Order, trade_val: float, cost: float, trade_price: float) -> None:
        """更新杠杆订单"""
        # 基本的杠杆订单处理，可以根据需要扩展
        stock_id = order.stock_id
        
        if stock_id not in self.position:
            self._init_leveraged_stock(stock_id, leverage=getattr(order, 'leverage', self.default_leverage))
        
        if order.direction in [Order.BUY_LONG, Order.BUY]:
            self._update_long_position(stock_id, trade_val, cost, trade_price, True)
        elif order.direction in [Order.SELL_LONG, Order.SELL]:
            self._update_long_position(stock_id, trade_val, cost, trade_price, False)
        elif order.direction == Order.SELL_SHORT:
            self._update_short_position(stock_id, trade_val, cost, trade_price, True)
        elif order.direction == Order.BUY_SHORT:
            self._update_short_position(stock_id, trade_val, cost, trade_price, False)

    def _update_long_position(self, stock_id: str, trade_val: float, cost: float, trade_price: float, is_open: bool) -> None:
        """更新多头仓位"""
        trade_amount = trade_val / trade_price
        stock_pos = self.position[stock_id]
        
        if is_open:  # 开多仓
            stock_pos['long_amount'] += trade_amount
            self.position['cash'] -= trade_val + cost
        else:  # 平多仓
            stock_pos['long_amount'] -= trade_amount
            self.position['cash'] += trade_val - cost
            
        stock_pos['net_amount'] = stock_pos['long_amount'] - stock_pos['short_amount']
        stock_pos['price'] = trade_price
        stock_pos['total_cost'] += cost

    def _update_short_position(self, stock_id: str, trade_val: float, cost: float, trade_price: float, is_open: bool) -> None:
        """更新空头仓位"""
        trade_amount = trade_val / trade_price
        stock_pos = self.position[stock_id]
        
        if is_open:  # 开空仓
            stock_pos['short_amount'] += trade_amount
            self.position['cash'] += trade_val - cost
        else:  # 平空仓
            stock_pos['short_amount'] -= trade_amount
            self.position['cash'] -= trade_val + cost
            
        stock_pos['net_amount'] = stock_pos['long_amount'] - stock_pos['short_amount']
        stock_pos['price'] = trade_price
        stock_pos['total_cost'] += cost

    def update_stock_price(self, stock_id: str, price: float) -> None:
        if stock_id in self.position:
            self.position[stock_id]['price'] = price

    def calculate_stock_value(self) -> float:
        """计算股票价值（净仓位价值）"""
        value = 0
        for stock_id, pos_data in self.position.items():
            if stock_id not in ['cash', 'total_margin', 'total_equity']:
                net_amount = pos_data['net_amount']
                price = pos_data['price']
                value += net_amount * price
        return value

    def calculate_value(self) -> float:
        """计算总价值"""
        return self.calculate_stock_value() + self.position['cash']

    def get_stock_list(self) -> List[str]:
        return [k for k in self.position.keys() if k not in ['cash', 'total_margin', 'total_equity']]

    def get_stock_price(self, code: str) -> float:
        return self.position[code]['price'] if code in self.position else 0.0

    def get_stock_amount(self, code: str) -> float:
        """返回净仓位（多头-空头）"""
        return self.position[code]['net_amount'] if code in self.position else 0.0

    def get_cash(self, include_settle: bool = False) -> float:
        return self.position['cash']

    def get_stock_amount_dict(self) -> dict:
        """生成股票数量字典"""
        d = {}
        for stock_id in self.get_stock_list():
            d[stock_id] = self.get_stock_amount(stock_id)
        return d

    def get_stock_weight_dict(self, only_stock: bool = False) -> dict:
        """生成股票权重字典"""
        if only_stock:
            total_value = self.calculate_stock_value()
        else:
            total_value = self.calculate_value()
            
        if total_value == 0:
            return {}
            
        d = {}
        for stock_id in self.get_stock_list():
            net_amount = self.position[stock_id]['net_amount']
            price = self.position[stock_id]['price']
            d[stock_id] = (net_amount * price) / total_value
        return d

    def add_count_all(self, bar: str) -> None:
        """增加持有时间计数"""
        for stock_id in self.get_stock_list():
            self.position[stock_id]['count'] += 1

    def update_weight_all(self) -> None:
        """更新所有权重"""
        weights = self.get_stock_weight_dict()
        for stock_id, weight in weights.items():
            self.position[stock_id]['weight'] = weight

    def settle_start(self, settle_type: str) -> None:
        """开始结算"""
        pass

    def settle_commit(self) -> None:
        """提交结算"""
        pass

    # 杠杆相关方法
    def get_total_margin(self) -> float:
        """获取总保证金"""
        return self.position.get('total_margin', 0.0)

    def get_total_unrealized_pnl(self) -> float:
        """获取总未实现盈亏"""
        total_pnl = 0.0
        for stock_id in self.get_stock_list():
            total_pnl += self.position[stock_id].get('unrealized_pnl', 0.0)
        return total_pnl

    def get_total_realized_pnl(self) -> float:
        """获取总已实现盈亏"""
        total_pnl = 0.0
        for stock_id in self.get_stock_list():
            total_pnl += self.position[stock_id].get('realized_pnl', 0.0)
        return total_pnl

    @property
    def long_positions(self) -> dict:
        """获取多头仓位"""
        return {k: v for k, v in self.position.items() 
                if k not in ['cash', 'total_margin', 'total_equity'] and v.get('long_amount', 0) > 0}

    @property
    def short_positions(self) -> dict:
        """获取空头仓位"""
        return {k: v for k, v in self.position.items() 
                if k not in ['cash', 'total_margin', 'total_equity'] and v.get('short_amount', 0) > 0}

    def get_available_cash(self) -> float:
        """获取可用现金"""
        return max(0, self.position['cash'] - self.get_total_margin())

    def get_leverage_ratio(self) -> float:
        """获取当前杠杆比率"""
        equity = self.calculate_value()
        if equity <= 0:
            return 0.0
        total_position_value = abs(self.calculate_stock_value())
        return total_position_value / equity if equity > 0 else 0.0
