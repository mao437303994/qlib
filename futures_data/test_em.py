import os
import re

import numpy as np
import requests
import pandas as pd
import datetime
from fees import futures_fees_info


def futures_jq_efi() -> pd.DataFrame:
    url = "https://futsseapi.eastmoney.com/list/risk/efi?orderBy=&sort=&pageSize=999&pageIndex=0&specificContract=true&platform=zbPC&field=name,dm,sc,uid"
    r = requests.get(url=url)
    data_json = r.json()

    sc = []
    dm = []
    name = []
    uid = []

    for item in data_json["list"]:
        sc.append(item["sc"])
        dm.append(item["dm"])
        name.append(item["name"])
        uid.append(item["uid"])

    return pd.DataFrame({"sc": sc, "dm": dm, "name": name, "uid": uid})


def futures_jq_em() -> pd.DataFrame:
    url = "https://futsseapi.eastmoney.com/list/trans/block/risk/mk0830?orderBy=&sort=&pageSize=999&pageIndex=0&specificContract=true&platform=zbPC&field=name,dm,sc,uid"
    r = requests.get(url=url)
    data_json = r.json()

    sc = []
    dm = []
    name = []
    uid = []

    for item in data_json["list"]:
        sc.append(item["sc"])
        dm.append(item["dm"])
        name.append(item["name"])
        uid.append(item["uid"])

    return pd.DataFrame({"sc": sc, "dm": dm, "name": name, "uid": uid})


def futures_jq_hist_em(symbol: str) -> pd.DataFrame:
    url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get"

    params = {
        "secid": symbol,
        "klt": "101",
        "fqt": "1",
        "lmt": "10000",
        "end": "20500000",
        "iscca": "1",
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "forcect": "1",
    }

    r = requests.get(url=url, params=params)
    data_json = r.json()
    code = data_json["data"]["code"]
    klines = data_json["data"]["klines"]

    date = []
    open = []
    close = []
    high = []
    low = []
    volume = []
    position = []

    for item in klines:
        kl = item.split(",")
        date.append(kl[0])
        open.append(kl[1])
        close.append(kl[2])
        high.append(kl[3])
        low.append(kl[4])
        volume.append(kl[5])
        position.append(kl[12])

    return pd.DataFrame(
        {
            "symbol": code,
            "date": date,
            "open": open,
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "position": position,
        }
    )


class Variety:

    def __init__(self, sc: str, dm: str, name: str):
        self.sc = sc
        self.dm = dm
        self.name = name

    def __hash__(self):
        return hash((self.sc, self.dm.upper()))

    def __eq__(self, other):
        if not isinstance(other, Variety):
            return NotImplemented
        if not self.sc == other.sc:
            return False
        if not self.dm.upper() == other.dm.upper():
            return False
        return True

    def __str__(self):
        return f"{self.sc}.{self.dm} ({self.name})"

    def __repr__(self):
        return f"Variety(sc={self.sc!r}, dm={self.dm!r}, name={self.name!r})"


def get_variety_list() -> pd.DataFrame:

    def to_efi(dm: str) -> str:
        m = re.match(r"([A-Z]+)\d+$", dm)
        if m:
            variety_code = m.group(1)
            return variety_code + "FI"
        else:
            m = re.match(r"([a-z]+)\d+$", dm)
            if m:
                variety_code = m.group(1)
                return variety_code + "fi"

        raise ValueError(f"无法从 '{dm}' 提取品种代码")

    _futures_jq_em = futures_jq_em()

    em_variety = {
        Variety("159", to_efi(row["dm"]), row["name"])
        for _, row in _futures_jq_em.iterrows()
        if not row["uid"].startswith("CFFEX")
    }

    _futures_jq_efi = futures_jq_efi()

    efi_variety = {
        Variety("159", row["dm"], row["name"])
        for _, row in _futures_jq_efi.iterrows()
        if not row["uid"].startswith("CFFEX")
    }

    varietys = efi_variety | em_variety  # 合并两个集合
    sc = [v.sc for v in varietys]
    dm = [v.dm for v in varietys]
    name = [v.name for v in varietys]

    return pd.DataFrame({"sc": sc, "dm": dm, "name": name})


if __name__ == "__main__":

    ## 下载指数日频数据
    dir = os.path.dirname(__file__)

    futures = os.path.join(dir, "futures/em.csv")
    if not os.path.exists(futures):
        os.makedirs(os.path.dirname(futures), exist_ok=True)
        _get_variety_list = get_variety_list()
        _get_variety_list.to_csv(futures, index=False)
    else:
        _get_variety_list = pd.read_csv(futures)

    fromPath = os.path.join(dir, "data")
    os.makedirs(fromPath, exist_ok=True)

    fees_df = futures_fees_info()

    for index, row in _get_variety_list.iterrows():
        filename = os.path.join(fromPath, f"{row['dm']}.csv")
        if not os.path.exists(filename):
            symbol = f"{row['sc']}.{row['dm']}"
            data = futures_jq_hist_em(symbol)

            m = re.match(r"([a-z]+)(fi|\d+)$", row["dm"], re.IGNORECASE)
            variety_code = m.group(1) if m else None

            if (variety_code is None) or (variety_code.upper() not in fees_df.index):
                contract_multipliers = [np.nan] * len(data)
                open_commission_rate = [np.nan] * len(data)
                open_commission_per_lot = [np.nan] * len(data)
                close_commission_rate = [np.nan] * len(data)
                close_commission_per_lot = [np.nan] * len(data)
                close_today_commission_rate = [np.nan] * len(data)
                close_today_commission_per_lot = [np.nan] * len(data)
                long_margin_rate = [np.nan] * len(data)
                short_margin_rate = [np.nan] * len(data)
            else:
                variety_info = fees_df.loc[variety_code.upper()]
                contract_multipliers = [variety_info["contract_multiplier"]] * len(data)
                open_commission_rate = [variety_info["open_commission_rate"]] * len(
                    data
                )
                open_commission_per_lot = [
                    variety_info["open_commission_per_lot"]
                ] * len(data)
                close_commission_rate = [variety_info["close_commission_rate"]] * len(
                    data
                )
                close_commission_per_lot = [
                    variety_info["close_commission_per_lot"]
                ] * len(data)
                close_today_commission_rate = [
                    variety_info["close_today_commission_rate"]
                ] * len(data)
                close_today_commission_per_lot = [
                    variety_info["close_today_commission_per_lot"]
                ] * len(data)
                long_margin_rate = [variety_info["long_margin_rate"]] * len(data)
                short_margin_rate = [variety_info["short_margin_rate"]] * len(data)

            date = pd.to_datetime(data["date"])
            temp = pd.DataFrame(
                {
                    "symbol": data["symbol"],
                    "date": date,
                    "open": data["open"].astype(float),
                    "close": data["close"].astype(float),
                    "high": data["high"].astype(float),
                    "low": data["low"].astype(float),
                    "volume": data["volume"].astype(float),
                    "change": data["close"].astype(float).pct_change().fillna(0),
                    "factor": 1.0,  # 假设没有复权
                    "oi": data["position"].astype(float),
                    "month": date.dt.month,
                    "week": date.dt.isocalendar().week,
                    "quarter": date.dt.quarter,
                    ## 添加手续费信息
                    "contract_multiplier": contract_multipliers,
                    "open_commission_rate": open_commission_rate,
                    "open_commission_per_lot": open_commission_per_lot,
                    "close_commission_rate": close_commission_rate,
                    "close_commission_per_lot": close_commission_per_lot,
                    "close_today_commission_rate": close_today_commission_rate,
                    "close_today_commission_per_lot": close_today_commission_per_lot,
                    "long_margin_rate": long_margin_rate,
                    "short_margin_rate": short_margin_rate,
                }
            )

            today = pd.Timestamp("today")
            if today.hour < 15:
                today = today.normalize()
                temp = temp[temp["date"] < today]  # 过滤掉未来数据

            ptc = temp["close"].pct_change().abs() * 100
            temp = temp[(ptc < 10) | (ptc.isna())]  # 过滤掉涨跌幅超过10%的数据
            temp = temp[temp["oi"] > 10000]  # 过滤掉持仓量小于1万的合约

            if len(temp) > 270 * 3:
                temp.to_csv(filename, index=False)

    import sys

    sys.path.insert(1, os.getcwd())

    import shutil
    from scripts.dump_bin import DumpDataAll

    toPath = "~/.qlib/qlib_data/cn_future"
    toPath = os.path.expanduser(toPath)
    os.makedirs(toPath, exist_ok=True)
    shutil.rmtree(toPath)

    DumpDataAll(
        csv_path=fromPath,
        qlib_dir=toPath,
        max_workers=1,
        include_fields="open,close,high,low,volume,oi,month,week,quarter,factor,change,contract_multiplier,open_commission_rate,open_commission_per_lot,close_commission_rate,close_commission_per_lot,close_today_commission_rate,close_today_commission_per_lot,long_margin_rate,short_margin_rate",
        date_field_name="date",
        symbol_field_name="symbol",
    ).dump()

    all = os.path.join(toPath, "instruments/all.txt")
    if os.path.exists(all):
        futures = pd.read_csv(
            all,
            sep="\t",
            header=None,
            names=["code", "start_date", "end_date"],
        )

        active = pd.read_csv(
            os.path.join(dir, "active.txt"),
            sep="\t",
            header=None,
            names=["code"],
        )

        futures_active = futures[futures["code"].isin(active["code"])].copy()
        futures_active.to_csv(
            os.path.join(toPath, "instruments/active.txt"),
            sep="\t",
            header=False,
            index=False,
        )

        emindfi = pd.read_csv(
            os.path.join(dir, "emindfi.txt"),
            sep="\t",
            header=None,
            names=["code"],
        )
        emindfi = futures[futures["code"].isin(emindfi["code"])].copy()
        emindfi.to_csv(
            os.path.join(toPath, "instruments/emindfi.txt"),
            sep="\t",
            header=False,
            index=False,
        )

    pass
