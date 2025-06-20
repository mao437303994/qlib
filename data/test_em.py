from requests import get
import pandas as pd
import os


def futures_exchange_em():
    url = "https://futsse-static.eastmoney.com/redis"
    r = get(url=url, params={"msgid": "gnweb"})
    data_json = r.json()
    mktid = []
    mktname = []
    mktshort = []
    for item in data_json:
        mktid.append(item["mktid"])
        mktname.append(item["mktname"])
        mktshort.append(item["mktshort"])

    return pd.DataFrame({"mktid": mktid, "mktname": mktname, "mktshort": mktshort})


def futures_pz_em(exchange: pd.DataFrame):
    url = "https://futsse-static.eastmoney.com/redis"

    mktid = []
    mktname = []
    mktshort = []
    vcode = []
    vname = []
    vtype = []

    for item in exchange["mktid"]:
        r = get(url=url, params={"msgid": str(item)})
        data_json = r.json()
        mktid.extend([item["mktid"] for item in data_json])
        mktname.extend([item["mktname"] for item in data_json])
        mktshort.extend([item["mktshort"] for item in data_json])
        vcode.extend([item["vcode"] for item in data_json])
        vname.extend([item["vname"] for item in data_json])
        vtype.extend([item["vtype"] for item in data_json])

    return pd.DataFrame(
        {
            "mktid": mktid,
            "mktname": mktname,
            "mktshort": mktshort,
            "vcode": vcode,
            "vname": vname,
            "vtype": vtype,
        }
    )


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


# python scripts/dump_bin.py dump_all --max_workers 1 --csv_path  E:/qh/data --qlib_dir ~/.qlib/qlib_data/cn_future --include_fields open,close,high,low,volume,position,month,day --date_field_name date --symbol_field_name symbol


if __name__ == "__main__":

    ## 下载指数日频数据
    dir = os.path.dirname(__file__)
    # _futures_jq_em = futures_jq_em()

    # for index, row in _futures_jq_em.iterrows():
    #     filename = os.path.join(dir, f"data/{row['dm']}.csv")
    #     if not os.path.exists(filename):
    #         symbol = f"{row['sc']}.{row['dm']}"
    #         data = futures_jq_hist_em(symbol)
    #         date = pd.to_datetime(data["date"])
    #         temp = pd.DataFrame(
    #             {
    #                 "symbol": data["symbol"],
    #                 "date": date,
    #                 "open": data["open"].astype(float),
    #                 "close": data["close"].astype(float),
    #                 "high": data["high"].astype(float),
    #                 "low": data["low"].astype(float),
    #                 "volume": data["volume"].astype(float),
    #                 "oi": data["position"].astype(float),
    #                 "timestamp": date.astype("int64") // 10**9,
    #             }
    #         )
    #         temp.to_csv(filename, index=False)

    ### 处理emind集合
    # df1 = pd.read_csv(
    #     "~/.qlib/qlib_data/cn_future/instruments/all.txt",
    #     sep="\t",
    #     header=None,
    #     names=["code", "start_date", "end_date"],
    # )

    # codes = []
    # start_times = []
    # end_times = []

    # dir = os.path.dirname(__file__)
    # df = pd.read_csv(
    #     os.path.join(dir, "futures_emind.txt"),
    #     sep="\t",
    #     header=None,
    #     names=["code"],
    # )

    # for index, row in df.iterrows():
    #     code = row["code"].upper()
    #     _row = df1[df1["code"] == code]

    #     codes.append(_row["code"].values[0])
    #     start_times.append(_row["start_date"].values[0])
    #     end_times.append(_row["end_date"].values[0])

    # t = pd.DataFrame(
    #     {
    #         "code": codes,
    #         "start_date": start_times,
    #         "end_date": end_times,
    #     }
    # )

    # t.to_csv(
    #     "~/.qlib/qlib_data/cn_future/instruments/emindfi.txt",
    #     index=False,
    #     header=False,
    #     sep="\t",
    # )

    fs = os.listdir("data/data")
    for f in fs:
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join("data/data", f))
            t = pd.DataFrame()
            b = pd.to_datetime(df["date"])
            t["symbol"] = df["symbol"]
            t["date"] = b
            t["open"] = df["open"].astype(float)
            t["close"] = df["close"].astype(float)
            t["high"] = df["high"].astype(float)
            t["low"] = df["low"].astype(float)
            t["volume"] = df["volume"].astype(float)
            t["oi"] = df["oi"].astype(float)
            t["month"] = b.dt.month
            t["week"] = b.dt.isocalendar().week  # 周数
            t["quarter"] = b.dt.quarter          # 季度
            t.to_csv(os.path.join("data/data/data", f), index=False)
    pass
