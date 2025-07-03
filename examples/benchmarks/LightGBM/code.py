from os import path
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.workflow.cli import workflow
from qlib.workflow import R
import qlib
from qlib.workflow.record_temp import PortAnaRecord
from qlib.data import D

if __name__ == "__main__":
    # path = path.join(path.dirname(__file__), "workflow_config_lightgbm_Alpha158.yaml")
    # workflow(config_path=path)

    # 761191810450610219
    # b9a09d2216804d8c9f7085e840a82a6c

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")
    # df = D.features(
    #     instruments=["SH601008"],
    #     fields=["$close", "$open", "$high", "$low", "$volume", "$change", "$factor"],
    #     start_time="2020-01-01",
    #     end_time="2020-01-05",
    # )
    # print(df)
    recorder = R.get_recorder(
        recorder_id="b9a09d2216804d8c9f7085e840a82a6c",
        experiment_id="761191810450610219",
    )
    pr = PortAnaRecord(
        recorder=recorder,
        config={
            "strategy": {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy",
                "kwargs": {"signal": "<PRED>", "topk": 50, "n_drop": 5},
            },
            "backtest": {
                "start_time": "2017-01-01",
                "end_time": "2020-08-01",
                "account": 10000000,
                # "benchmark": "emind",
                "benchmark": "SH000300",
                "pos_type": "FuturePosition",
                "exchange_kwargs": {
                    "trade_unit": 100,
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            },
            "executor": {
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "day",
                    "generate_portfolio_metrics": True,
                },
            },
        },
    )
    pr.generate()
