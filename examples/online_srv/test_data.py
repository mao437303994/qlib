import os
import pandas as pd

if __name__ == "__main__":
    df1 = pd.read_csv(
        "~/.qlib/qlib_data/cn_future/instruments/active.txt",
        sep="\t",
        header=None,
        names=["code", "start_date", "end_date"],
    )

    for index, row in df1.iterrows():
        code = row["code"].upper()

        if os.path.exists(f"./data/data/{code}.csv"):
            df = pd.read_csv(f"./data/data/{code}.csv")

            # # 检查缺失值
            # print(f"{code} 缺失值统计：")
            # print(df.isnull().sum().all())

            # # 检查数据类型
            # print(f"{code} 数据类型：")
            # print(df.dtypes)

            # # 检查极值和分布
            # print(f"{code} describe: ")
            # print(df.describe())

            # 涨跌幅
            df["close_pct"] = df["close"].pct_change()
            a = df[(df["close_pct"] > 0.105) | (df["close_pct"] < -0.105)]
            if len(a) > 0:
                print(f"{code} 涨跌幅异常：")
                print(a[["date", "close_pct"]])

            # 检查是否有重复行
            dup = df.duplicated(subset=["date"])
            if dup.any():
                print(f"{code} 存在重复日期：", df[dup])

            # 检查日期是否递增
            if not df["date"].is_monotonic_increasing:
                print(f"{code} 日期未递增！")

            print("=" * 40)
