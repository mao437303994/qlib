from alpha import Alpha
from qlib.contrib.data.handler import Alpha158
import qlib
from qlib.data import D
from qlib.data.dataset.processor import DropnaLabel, Processor, ZScoreNorm
from qlib.model.trainer import task_train
from qlib.utils.mod import init_instance_by_config
from qlib.workflow import R
from sar import SAR
import pandas as pd


if __name__ == "__main__":
    qlib.init(
        provider_uri="~/.qlib/qlib_data/cn_future",
        region="cn_future",
        kernels=1,
        **{"custom_ops": [SAR]},
    )

    # 查看某只股票的原始行情数据

    # instruments = D.instruments("active")

    # df = D.features(
    #     instruments,
    #     [
    #         "$close",
    #         "$open",
    #         "$high",
    #         "$low",
    #         "$volume",
    #         "$oi",
    #         "$month",
    #         "$day",
    #     ],
    # )

    handler = Alpha(
        instruments="acitve",
        start_time="2013-01-01",
        end_time="2025-05-22",
        learn_processors=[
            DropnaLabel(
                fields_group="label",
            ),
            ZScoreNorm(
                fields_group="feature",
                fit_start_time="2013-01-01",
                fit_end_time="2022-12-31",
            ),
        ],
    )

    df = handler.fetch(col_set=["feature", "label"])

    df["feature"].describe().to_csv("feature.csv")
    df["label"].describe().to_csv("label.csv")
    df.corr().to_csv("corr.csv")

    # df.to_csv("c.csv")

    # config = {
    #     "model": {
    #         "class": "LGBModel",
    #         "module_path": "qlib.contrib.model.gbdt",
    #         "kwargs": {
    #             "loss": "mse",
    #             "colsample_bytree": "0.8879",
    #             "learning_rate": "0.2",
    #             "subsample": "0.8789",
    #             "lambda_l1": "205.6999",
    #             "lambda_l2": "580.9768",
    #             "max_depth": "8",
    #             "num_leaves": "210",
    #             "num_threads": "20",
    #         },
    #     },
    #     "dataset": {
    #         "class": "DatasetH",
    #         "module_path": "qlib.data.dataset",
    #         "kwargs": {
    #             "handler": {
    #                 "class": "Alpha",
    #                 "module_path": "test",
    #                 "kwargs": {
    #                     "instruments": "active",
    #                     "start_time": "2013-01-01",
    #                     "end_time": "2025-05-22",
    #                     "learn_processors": [
    #                         DropnaLabel(fields_group="label"),
    #                         ZScoreNorm(
    #                             fit_start_time="2013-01-01",
    #                             fit_end_time="2022-12-31",
    #                             fields_group="feature",
    #                         ),
    #                     ],
    #                 },
    #             },
    #             "segments": {
    #                 "train": ("2013-01-01", "2020-12-31"),
    #                 "valid": ("2021-01-01", "2022-12-31"),
    #                 "test": ("2023-01-01", "2025-05-22"),
    #             },
    #         },
    #     },
    #     "record": {
    #         "class": "SignalRecord",
    #         "module_path": "qlib.workflow.record_temp",
    #         "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"},
    #     },
    # }

    # task_train(config, experiment_name="future_alpha_test")
