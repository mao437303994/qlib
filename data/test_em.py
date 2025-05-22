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


if __name__ == "__main__":

    # futures_exchange_em = futures_exchange_em()
    # futures_exchange_em.to_csv("futures_exchange_em.csv", index=False)
    # futures_exchange_em = pd.read_csv("futures_exchange_em.csv")
    # futures_pz_em = futures_pz_em(futures_exchange_em)
    # futures_pz_em.to_csv("futures_pz_em.csv", index=False)

    # futures_jq_em = futures_jq_em()
    # futures_jq_em.to_csv("futures_jq_em.csv", index=False)

    # futures_jq_em_data = pd.read_csv("futures_jq_em.csv")
    # for index, row in futures_jq_em_data.iterrows():
    #     filename = f"./data/{row['dm']}.csv"
    #     if not os.path.exists(filename):
    #         symbol = f"{row['sc']}.{row['dm']}"
    #         data = futures_jq_hist_em(symbol)
    #         data.to_csv(filename, index=False)

    df1 = pd.read_csv(
        "c:\\Users\\admin\\.qlib\\qlib_data\\cn_future\\instruments\\all.txt",
        sep="\t",
        header=None,
        names=["code", "start_date", "end_date"],
    )

    codes = []
    start_times = []
    end_times = []

    df = pd.read_csv("futures_emind_cf_em.csv")
    for index, row in df.iterrows():
        code = row["code"].upper()
        _row = df1[df1["code"] == code]

        codes.append(_row["code"].values[0])
        start_times.append(_row["start_date"].values[0])
        end_times.append(_row["end_date"].values[0])

    t = pd.DataFrame(
        {
            "code": codes,
            "start_date": start_times,
            "end_date": end_times,
        }
    )

    t.to_csv(
        "c:\\Users\\admin\\.qlib\\qlib_data\\cn_future\\instruments\\emindfi.txt",
        index=False,
        header=False,
        sep="\t",
    )
