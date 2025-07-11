from os import path
from qlib.workflow.cli import workflow
from qlib.workflow import R
import qlib
from qlib.workflow.record_temp import PortAnaRecord 

if __name__ == "__main__":
    # path = path.join(path.dirname(__file__), "workflow_config_lightgbm_Alpha158.yaml")
    # workflow(config_path=path)
    # 22f752445ca7433dab7d5d2ec45f0bab
    # 324285785865736158

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

    r = R.get_recorder(experiment_id="837853288875549555",recorder_id="945dba5f893b42028d262c1d132dab0b")
    pr = PortAnaRecord( recorder=r ,config={
        "executor":{
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {"signal": "<PRED>", "topk": 50, "n_drop": 5},
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": "SH000300",
            "exchange_kwargs": {
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    })

    pr.generate()