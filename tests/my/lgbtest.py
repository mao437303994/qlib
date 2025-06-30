from alpha import Alpha
from dataset import DatasetS
from ops import Month, Day, SAR, Sin, Cos
from qlib.config import DISK_DATASET_CACHE, DISK_EXPRESSION_CACHE
from qlib.contrib.data.handler import Alpha158
import qlib
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

# import sweetviz as sv
# from ydata_profiling import ProfileReport
import optuna
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH
import lightgbm as lgb
from lightgbm import LGBMRegressor
from processors import Clip

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

if __name__ == "__main__":
    qlib.init(
        provider_uri="~/.qlib/qlib_data/cn_future",
        region="cn_future",
        kernels=1,
        **{"custom_ops": [Month, Day, Sin, Cos]},
    )

    # filterCols = FilterCol(col_list=["VSUMP20"])

    filterCols = DropCol(col_list=["VWAP0"])

    handler = Alpha(
        # drop_raw=True,
        # instruments="csi300",
        instruments="active",
        start_time="2006-01-01",
        end_time="2025-05-22",
        process_type=DataHandlerLP.PTYPE_I,
        shared_processors=[filterCols],
        infer_processors=[
            ZScoreNorm(
                fields_group="feature",
                fit_start_time="2006-01-01",
                fit_end_time="2017-08-14",
            ),
        ],
        learn_processors=[
            DropnaLabel(fields_group="label"),
            DropnaLabel(fields_group="feature"),
            ZScoreNorm(
                fields_group="feature",
                fit_start_time="2006-01-01",
                fit_end_time="2017-08-14",
            ),
        ],
    )

    # df = handler.fetch(col_set=["feature", "label"], data_key="learn")
    # df["label"].to_csv("label.csv")
    # df1 = handler.fetch(col_set=["feature", "label"], data_key="raw")
    # 生成分析报告
    # report = sv.analyze(df["feature"])
    # 保存为HTML文件
    # report.show_html("sweetviz_report_1.html")

    # profile = ProfileReport(df["feature"], title="特征分析报告", explorative=True)
    # # 保存为HTML文件
    # profile.to_file("profile_report.html")

    # profile = ProfileReport(df["label"], title="标签分析报告", explorative=True)
    # 保存为HTML文件
    # profile.to_file("label_profile_report.html")

    # df.to_csv("feature.csv")
    # df["feature"].describe().to_csv("feature_describe.csv")
    # df["label"].describe().to_csv("label_describe.csv")
    # df.corr().to_csv("corr.csv")

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": ("2006-01-01", "2017-08-14"),
            "valid": ("2017-08-15", "2021-06-29"),
            "test": ("2021-06-30", "2025-05-22"),
        },
    )

    def objective(trial):

        param = {
            "lose": "binary",
            "boosting": "gbdt",
            "metric": "auc",
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        }

        model = LGBModel(**param)
        evals_result = {}
        model.fit(dataset=dataset, evals_result=evals_result)
        return max(evals_result["valid"]["auc"])

    study = optuna.create_study(
        study_name="my_study",
        direction="maximize",  # ["minimize", "maximize"],
    )

    study.optimize(objective, n_trials=50)

    a = []
    b = {}

    for k, v in study.best_params.items():
        _index = k.find("use_")  # Check if the key starts with "use_"
        if _index > -1:
            if study.best_params[k] is True:
                a.append(k[_index + 4 :])  # Extract the feature name
        else:
            b.update({k: v})

    print("Best params:", a)
    print("Best params:", b)

    # df = study.trials_dataframe()
    # df.to_csv("optuna_trials.csv", index=False)

    # fig = optuna.visualization.plot_optimization_history(study)
    # fig.write_html("plot_optimization_history.html")
    # fig = optuna.visualization.plot_param_importances(study)
    # fig.write_html("plot_param_importances.html")
    # fig = optuna.visualization.plot_parallel_coordinate(study)
    # fig.write_html("plot_parallel_coordinate.html")

    config = {
        "dataset": dataset,
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "learning_rate": 0.0540573778890074,
                "num_leaves": 47,
                "max_depth": 3,
                "lambda_l1": 0.005448745443420065,
                "lambda_l2": 4.2068022977633275,
                "feature_fraction": 0.9175216653656415,
                "bagging_fraction": 0.9108460795811167,
                "bagging_freq": 1,
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
        ],
    }

    # task_train(config, experiment_name="future_alpha_test")
    # exp = R.get_exp(experiment_name="future_alpha_test")
    # r =exp.get_recorder()
    # r.load_object("pre")
