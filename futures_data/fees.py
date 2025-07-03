from io import StringIO
import re
import pandas as pd
import requests


def futures_fees_info() -> pd.DataFrame:
    url = "http://openctp.cn/fees.html"
    r = requests.get(url)
    r.encoding = "utf-8"
    temp_df, *_ = pd.read_html(StringIO(r.text))
    temp_df = temp_df.drop_duplicates(subset=["品种代码"])
    temp_df = temp_df.reset_index(drop=True)

    df = pd.DataFrame(
        {
            "variety_code": temp_df["品种代码"],
            "variety_name": temp_df["品种名称"],
            "contract_multiplier": temp_df["合约乘数"],
            "min_tick": temp_df["最小跳动"],
            "open_commission_rate": temp_df["开仓费率（按金额）"],
            "open_commission_per_lot": temp_df["开仓费用（按手）"],
            "close_commission_rate": temp_df["平仓费率（按金额）"],
            "close_commission_per_lot": temp_df["平仓费用（按手）"],
            "close_today_commission_rate": temp_df["平今费率（按金额）"],
            "close_today_commission_per_lot": temp_df["平今费用（按手）"],
            "long_margin_rate": temp_df["做多保证金率（按金额）"],
            "long_margin_per_lot": temp_df["做多保证金（按手）"],
            "short_margin_rate": temp_df["做空保证金率（按金额）"],
            "short_margin_per_lot": temp_df["做空保证金（按手）"],
        }
    )

    df.index = df["variety_code"].str.upper()

    return df


if __name__ == "__main__":
    all = pd.read_csv(
        "~/.qlib/qlib_data/cn_future/instruments/all.txt",
        sep="\t",
        header=None,
        names=[
            "symbol",
            "start_date",
            "end_date",
        ],
    )

    fees_df = futures_fees_info()

    for index, row in all.iterrows():
        symbol = row["symbol"]
        m = re.match(r"([a-z]+)(fi|\d+)$", symbol, re.IGNORECASE)
        variety_code = m.group(1) if m else None
        if variety_code is None:
            print("指数: ", symbol)
        elif variety_code.upper() not in fees_df.index:
            print("不存在: ", symbol)
        else:
            variety_info = fees_df.loc[variety_code]
            # print(symbol)
