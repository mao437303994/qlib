
from qlib.data import D
import qlib
from sar import SAR

if __name__ == "__main__":

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data",
              region="cn", **{"custom_ops": [SAR]})

    df = D.features(
        instruments=["SH601008"],
        fields=["$close", "SAR(4,2,20)"],
        start_time="2020-01-01",
        end_time="2020-01-10",
    )

    print(df)
