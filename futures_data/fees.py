from io import StringIO
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
        },
    )

    df.index = df["variety_code"].str.upper()  # 将品种代码转换为大写并设为索引

    return df