from alpha import Alpha
from qlib.config import DISK_DATASET_CACHE, DISK_EXPRESSION_CACHE
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
        provider_uri="~/.qlib/qlib_data/cn_data",
        region="cn",
        kernels=1,
        # redis_host="127.0.0.1",
        # redis_port=6379,
        expression_cache=DISK_EXPRESSION_CACHE,
        dataset_cache=DISK_DATASET_CACHE,
        **{"custom_ops": [SAR]},
    )

    handler = Alpha(
        instruments="csi300",
        start_time="2006-05-29",
        end_time="2020-09-22",
    )

    df = handler.fetch(col_set=["feature", "label"])

    df["feature"].describe().to_csv("feature.csv")
    df["label"].describe().to_csv("label.csv")
    df.corr().to_csv("corr.csv")

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
    #                 "module_path": "alpha",
    #                 "kwargs": {
    #                     "instruments": "csi300",
    #                     "start_time": "2006-05-29",
    #                     "end_time": "2020-09-22",
    #                     # "learn_processors": [
    #                     #     # DropnaLabel(fields_group="label"),
    #                     #     # ZScoreNorm(
    #                     #     #     fit_start_time="2013-01-01",
    #                     #     #     fit_end_time="2022-12-31",
    #                     #     #     fields_group="feature",
    #                     #     # ),
    #                     # ],
    #                 },
    #             },
    #             "segments": {
    #                 "train": ("2006-05-29", "2014-05-29"),
    #                 "valid": ("2014-05-30", "2017-05-30"),
    #                 "test": ("2017-05-31", "2020-09-22"),
    #             },
    #         },
    #     },
    #     "record": {
    #         "class": "SignalRecord",
    #         "module_path": "qlib.workflow.record_temp",
    #         "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"},
    #     },
    # }

    # task_train(config, experiment_name="stock_alpha_test")
