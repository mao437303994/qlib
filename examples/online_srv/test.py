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


def plot_distribution(data, title="分布图", xlabel="值", bins=50, save_path=None):
    """
    画出数据的直方图和核密度估计图
    :param data: 一维数据（pandas Series 或 numpy array）
    :param title: 图标题
    :param xlabel: x轴标签
    :param bins: 直方图分箱数
    :param save_path: 如果不为None，则保存图片到该路径
    """
    plt.figure(figsize=(10, 6))
    # 画直方图和KDE
    sns.histplot(data, bins=bins, kde=True, color="skyblue", edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("频数/密度")
    plt.grid(True, linestyle="--", alpha=0.5)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()


def compare_distributions(train_label, predictions):
    """比较训练集标签和预测结果的分布"""
    # 创建图形
    plt.figure(figsize=(15, 10))

    # 确保数据对齐
    common_index = train_label.index.intersection(predictions.index)
    aligned_label = train_label.loc[common_index]
    aligned_predictions = predictions.loc[common_index]

    print(f"\n数据对齐信息:")
    print(f"训练集标签数量: {len(train_label)}")
    print(f"预测结果数量: {len(predictions)}")
    print(f"对齐后数量: {len(common_index)}")

    # 1. 绘制分布对比图
    plt.subplot(211)
    sns.kdeplot(
        data=aligned_label.values.flatten(),
        label="Training Labels",
        color="blue",
        alpha=0.6,
    )
    sns.kdeplot(
        data=aligned_predictions.values.flatten(),
        label="Predictions",
        color="red",
        alpha=0.6,
    )
    plt.title("Distribution Comparison")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()

    # 2. 绘制箱型图对比
    plt.subplot(212)
    plot_data = pd.DataFrame(
        {
            "Training Labels": aligned_label.values.flatten(),
            "Predictions": aligned_predictions.values.flatten(),
        }
    )
    sns.boxplot(data=plot_data)
    plt.title("Boxplot Comparison")

    plt.tight_layout()
    plt.savefig("distribution_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 计算统计信息
    stats_df = plot_data.describe()
    print("\n=== 分布统计对比 ===")
    print(stats_df)

    return stats_df


def compare_feature_distributions(
    raw_df,
    processed_df,
    feature_cols=None,
    sample_num=16,
    save_path="feature_distribution_comparison.png",
):
    """
    对比原始特征和处理后特征的分布
    :param raw_df: 原始特征DataFrame，格式df1["feature"]
    :param processed_df: 处理后特征DataFrame，格式df["feature"]
    :param feature_cols: 要对比的特征名列表，默认随机选sample_num个
    :param sample_num: 随机展示的特征数量
    :param save_path: 图片保存路径
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    if feature_cols is None:
        feature_cols = list(raw_df.columns)
        if len(feature_cols) > sample_num:
            feature_cols = np.random.choice(feature_cols, sample_num, replace=False)

    n = len(feature_cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=(ncols * 4, nrows * 3))

    for i, col in enumerate(feature_cols):
        plt.subplot(nrows, ncols, i + 1)
        sns.kdeplot(raw_df[col].dropna(), label="原始", color="blue", alpha=0.5)
        sns.kdeplot(processed_df[col].dropna(), label="处理后", color="red", alpha=0.5)
        plt.title(col)
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"特征分布对比图已保存到: {save_path}")


if __name__ == "__main__":
    qlib.init(
        provider_uri="~/.qlib/qlib_data/cn_future",
        region="cn_future",
        kernels=1,
        expression_cache=DISK_EXPRESSION_CACHE,
        dataset_cache=DISK_DATASET_CACHE,
        **{"custom_ops": [SAR]},
    )

    # a = DropCol(col_list=["VWAP0"])
    a = FilterCol(col_list=["MA5", "MA10", "MA20", "MA30", "MA60"])

    handler = Alpha(
        # instruments=["FGFI"],
        instruments="active",
        start_time="2013-01-01",
        end_time="2025-05-22",
        infer_processors=[
            a,
            # DropnaLabel(fields_group="label"),
            # DropnaLabel(fields_group="feature"),
            RobustZScoreNorm(
                fields_group="feature",
                fit_start_time="2013-01-01",
                fit_end_time="2021-12-31",
                clip_outlier=True,
            ),
        ],
        learn_processors=[
            a,
            DropnaLabel(fields_group="label"),
            DropnaLabel(fields_group="feature"),
            RobustZScoreNorm(
                fields_group="feature",
                fit_start_time="2013-01-01",
                fit_end_time="2021-12-31",
                clip_outlier=False,
            ),
            CSZScoreNorm(fields_group="label", method="robust"),
        ],
    )

    df = handler.fetch(col_set=["feature", "label"], data_key="learn")
    df1 = handler.fetch(col_set=["feature", "label"], data_key="raw")

    df  # 处理后的特征
    df1  # 原始特征

    compare_feature_distributions(
        df1["feature"],
        df["feature"],
        feature_cols=["MA5", "MA10", "MA20", "MA30", "MA60"],
    )

    # df.to_csv("feature.csv")
    # df["feature"].describe().to_csv("feature_describe.csv")
    # df["label"].describe().to_csv("label_describe.csv")
    # df.corr().to_csv("corr.csv")

    # config = {
    #     "model": {
    #         "class": "LGBModel",
    #         "module_path": "qlib.contrib.model.gbdt",
    #         "kwargs": {
    #             "objective": "regression_l2",
    #             "learning_rate": 0.005,  # 降低学习率以获得更稳定的训练过程
    #             "n_estimators": 2000,  # 增加树的数量，给模型更多学习机会
    #             "max_depth": 8,  # 适当增加深度，允许学习更复杂的特征
    #             "num_leaves": 50,  # 增加叶子节点，提高模型容量
    #             "feature_fraction": 0.9,  # 增加特征采样比例
    #             "bagging_fraction": 0.9,  # 增加样本采样比例
    #             "bagging_freq": 10,  # 调整采样频率
    #             "reg_alpha": 0.05,  # 减小L1正则化，允许更多特征参与
    #             "reg_lambda": 0.5,  # 减小L2正则化
    #             "min_child_samples": 10,  # 减小最小样本数要求
    #             "early_stopping_rounds": 200,  # 增加早停轮数，给模型更多机会
    #             "verbose": 50,
    #             "metric": ["l1", "l2"],
    #             # "first_metric_only": False,  # 同时监控两个指标
    #         },
    #     },
    #     "dataset": {
    #         "class": "DatasetH",
    #         "module_path": "qlib.data.dataset",
    #         "kwargs": {
    #             "handler": handler,
    #             "segments": {
    #                 "train": ("2013-01-01", "2019-12-31"),  # 缩短训练集
    #                 "valid": ("2020-01-01", "2021-12-31"),  # 增加验证集
    #                 "test": ("2022-01-01", "2023-12-31"),  # 缩短测试集
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

    # exp = R.get_exp(experiment_name="future_alpha_test")
    # recorder = exp.get_recorder()
    # score = recorder.load_object("pred.pkl")

    # score.to_csv("score.csv")
    # metrics = evaluate_predictions(score)
    # # 在训练后添加分析
    # score = recorder.load_object("pred.pkl")
    # analyze_predictions(score)

    # 比较分布
    # stats = compare_distributions(df["label"], score)
    # plot_distribution(score["score"], title="得分", xlabel="score")
    # plot_distribution(df["label"], title="Label distribution", xlabel="label")
