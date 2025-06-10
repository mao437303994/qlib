from alpha import Alpha
from qlib.config import DISK_DATASET_CACHE, DISK_EXPRESSION_CACHE
from qlib.contrib.data.handler import Alpha158
import qlib
from qlib.data import D
from qlib.data.dataset.processor import (
    DropnaLabel,
    ZScoreNorm,
    CSZScoreNorm,
    FilterCol,
    DropCol,
    ProcessInf,
    RobustZScoreNorm,
)
from qlib.model.trainer import task_train
from qlib.utils.mod import init_instance_by_config
from qlib.workflow import R
from sar import SAR
import pandas as pd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sweetviz as sv

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 或 ['Microsoft YaHei']
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题

if __name__ == "__main__":
    qlib.init(
        provider_uri="~/.qlib/qlib_data/cn_data",
        region="cn",
        kernels=1,
        # redis_host="127.0.0.1",
        # redis_port=6379,
        expression_cache=DISK_EXPRESSION_CACHE,
        dataset_cache=DISK_DATASET_CACHE,
        **{"custom_ops": [SAR]},
    )

    a = FilterCol(col_list=["MA5", "MA10", "MA20", "MA30", "MA60"])

    handler = Alpha(
        instruments="csi300",
        start_time="2006-05-29",
        end_time="2020-09-22",
        infer_processors=[
            a,
            # RobustZScoreNorm(
            #     fields_group="feature",
            #     fit_start_time="2013-01-01",
            #     fit_end_time="2021-12-31",
            #     clip_outlier=True,
            # ),
        ],
        learn_processors=[
            a,
            DropnaLabel(fields_group="label"),
            DropnaLabel(fields_group="feature"),
            # RobustZScoreNorm(
            #     fields_group="feature",
            #     fit_start_time="2013-01-01",
            #     fit_end_time="2021-12-31",
            #     clip_outlier=False,
            # ),
            CSZScoreNorm(fields_group="label", method="robust"),
        ],
    )

    # df = handler.fetch(col_set=["feature", "label"])

    config = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": "0.8879",
                "learning_rate": "0.2",
                "subsample": "0.8789",
                "lambda_l1": "205.6999",
                "lambda_l2": "580.9768",
                "max_depth": "8",
                "num_leaves": "210",
                "num_threads": "20",
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": handler,
                "segments": {
                    "train": ("2006-05-29", "2014-05-29"),
                    "valid": ("2014-05-30", "2017-05-30"),
                    "test": ("2017-05-31", "2020-09-22"),
                },
            },
        },
        "record": [
            {
                "class": "SignalRecord",
                "module_path": "qlib.workflow.record_temp",
                "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"},
            },
            {
                "class": "SigAnaRecord",
                "module_path": "qlib.workflow.record_temp",
                "kwargs": {"ana_long_short": False, "ann_scaler": 252},
            },
            {
                "class": "PortAnaRecord",
                "module_path": "qlib.workflow.record_temp",
                "kwargs": {
                    "config": {
                        "strategy": {
                            "class": "TopkDropoutStrategy",
                            "module_path": "qlib.contrib.strategy",
                            "kwargs": {
                                "signal": "<PRED>",
                                "topk": 50,
                                "n_drop": 5,
                            },
                        },
                        "backtest": {
                            "start_time": "2017-05-31",
                            "end_time": "2020-09-22",
                            "account": 1_000_000,
                            "benchmark": "SH000300",
                            "exchange_kwargs": {
                                "limit_threshold": 0.095,
                                "deal_price": "close",
                                "open_cost": 0.0005,
                                "close_cost": 0.0015,
                                "min_cost": 5,
                            },
                        },
                    }
                },
            },
        ],
    }

    #task_train(config, experiment_name="stock_alpha_test")
