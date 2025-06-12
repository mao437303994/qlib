from alpha import Alpha
from dataset import DatasetS
from ops import Month, Day, SAR, Sin, Cos
from qlib.config import DISK_DATASET_CACHE, DISK_EXPRESSION_CACHE
from qlib.contrib.data.handler import Alpha158
import qlib
from qlib.contrib.model.pytorch_lstm_ts import LSTM
from qlib.data import D

from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import QlibDataLoader
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
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import sweetviz as sv
from ydata_profiling import ProfileReport
import optuna
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset import TSDatasetH
import lightgbm as lgb
from lightgbm import LGBMRegressor
from processors import Clip
import torch.nn.functional as F
import torch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False



if __name__ == "__main__":
    qlib.init(
        provider_uri="~/.qlib/qlib_data/cn_future",
        region="cn_future",
        kernels=1,
        expression_cache=DISK_EXPRESSION_CACHE,
        dataset_cache=DISK_DATASET_CACHE,
        **{"custom_ops": [SAR, Month, Day, Sin, Cos]},
    )

    filterCols = DropCol(col_list=["VWAP0"])

    handler = Alpha(
        # instruments=["FGFI"],
        drop_raw=True,
        instruments="active",
        start_time="2006-01-01",
        end_time="2025-05-22",
        process_type=DataHandlerLP.PTYPE_I,
        shared_processors=[filterCols],
        infer_processors=[
            ZScoreNorm(
                fields_group="feature",
                fit_start_time="2006-01-01",
                fit_end_time="2021-06-29",
                # clip_outlier=(3, -3),
            ),
        ],
        learn_processors=[
            DropnaLabel(fields_group="label"),
            DropnaLabel(fields_group="feature"),
            Clip(col_list=["RET1"], clip_outlier=(0.05, -0.05)),
            ZScoreNorm(
                fields_group="feature",
                fit_start_time="2006-01-01",
                fit_end_time="2021-06-29",
                # clip_outlier=(3, -3),
            ),
        ],
    )

    dataset = TSDatasetH(
        handler=handler,
        segments={
            "train": ("2006-01-01", "2017-08-14"),
            "valid": ("2017-08-15", "2021-06-29"),
            "test": ("2021-06-30", "2025-05-22"),
        },
    )

    def objective(trial):
        # 超参数空间
        hidden_size = trial.suggest_int("hidden_size", 16, 128)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        n_epochs = trial.suggest_int("n_epochs", 20, 100)
        early_stop = trial.suggest_int("early_stop", 5, 20)

        # 获取特征数
        _, d_feat = dataset.handler.fetch(col_set=["feature"], data_key="learn").shape

        model = LSTM(
            d_feat=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            early_stop=early_stop,
            loss="bce",  # 二分类
            optimizer="adam",
            GPU=0,
        )

        a = pd.DataFrame()

        # 训练
        evals_result = {}
        model.fit(dataset, evals_result=evals_result)

        # 验证集评估
        valid_pred = model.predict(dataset)
        valid_label = dataset.prepare("test", col_set=["label"], data_key="infer")
        y_true = valid_label.values.squeeze()
        y_pred_prob = valid_pred.values.squeeze()
        y_pred = (y_pred_prob > 0.5).astype(int)

        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_prob)
        f1 = f1_score(y_true, y_pred)

        print(f"ACC: {acc}, AUC: {auc}, F1: {f1}")

        return 1 - acc  # 或 -f1, 1-auc，按你的优化目标

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Best params:", study.best_params)
