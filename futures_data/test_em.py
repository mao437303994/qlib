import os

from requests import get
import pandas as pd
import datetime


def futures_jq_em() -> pd.DataFrame:
    url = "https://futsseapi.eastmoney.com/list/risk/efi?orderBy=&sort=&pageSize=999&pageIndex=0&specificContract=true&platform=zbPC&field=name,dm,sc"
    r = get(url=url)
    data_json = r.json()

    sc = []
    dm = []
    name = []

    for item in data_json["list"]:
        sc.append(item["sc"])
        dm.append(item["dm"])
        name.append(item["name"])

    return pd.DataFrame({"sc": sc, "dm": dm, "name": name})


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

    r = get(url=url, params=params)
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


if __name__ == "__main__":

    ## 下载指数日频数据
    dir = os.path.dirname(__file__)

    futures = os.path.join(dir, "futures/em.csv")
    if not os.path.exists(futures):
        os.makedirs(os.path.dirname(futures), exist_ok=True)
        _futures_jq_em = futures_jq_em()
        _futures_jq_em.to_csv(futures, index=False)
    else:
        _futures_jq_em = pd.read_csv(futures)

    fromPath = os.path.join(dir, "data")
    os.makedirs(fromPath, exist_ok=True)

    for index, row in _futures_jq_em.iterrows():
        filename = os.path.join(fromPath, f"{row['dm']}.csv")
        if not os.path.exists(filename):
            symbol = f"{row['sc']}.{row['dm']}"
            data = futures_jq_hist_em(symbol)

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
                    "change": data["close"].pct_change().fillna(0),
                    "factor": 1.0,  # 假设没有复权
                    "oi": data["position"].astype(float),
                    "month": date.dt.month,
                    "week": date.dt.isocalendar().week,
                    "quarter": date.dt.quarter,
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
    shutil.rmtree(toPath)

    DumpDataAll(
        csv_path=fromPath,
        qlib_dir=toPath,
        max_workers=1,
        include_fields="open,close,high,low,volume,oi,month,week,quarter,factor,change",
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
