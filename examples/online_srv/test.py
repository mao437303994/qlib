from qlib.contrib.data.handler import Alpha158
import qlib
from qlib.data import D
from qlib.data.dataset.processor import DropnaLabel, Processor, ZScoreNorm
from qlib.model.trainer import task_train
from qlib.utils.mod import init_instance_by_config
from qlib.workflow import R
from sar import SAR


class Alpha(Alpha158):
    def get_feature_config(self):
        fields, names = super().get_feature_config()
        #fields += ["SAR(4,2,20)"]
        #names += ["SAR"]

        fields += ["$oi", "$oi/Ref($oi,1)-1"]
        names += ["OI", "OI_CHG"]

        fields += ["$month", "$day"]
        names += ["MONTH", "DAY"]

        return fields, names


if __name__ == "__main__":
    qlib.init(
        provider_uri="~/.qlib/qlib_data/cn_future",
        # provider_uri="~/.qlib/qlib_data/cn_data",
        region="cn",
        kernels=1,
        **{"custom_ops": [SAR]},
    )
    # handler = Alpha158(
    #     start_time="2017-12-28",
    #     end_time="2018-01-05",
    #     # fit_start_time="2018-09-10",
    #     # fit_end_time="2018-10-31",
    #     instruments=["SH600018"],
    # )

    # df = handler.fetch(col_set=["feature", "label"])
    # print(df.head())

    # # 查看某只股票的原始行情数据
    # df = D.features(
    #     ["SH600018"],
    #     ["$open", "$high", "$low", "$close", "$volume","$factor"],
    #     start_time="2017-12-28",
    #     end_time="2018-01-05"
    # )
    # df["original_close_price"] =df["$close"] / df["$factor"]
    # print(df)

    # 查看某只股票的原始行情数据
    # df = D.features(
    #     ["FGFI"],
    #     [
    #         "$close",
    #         "Ref($close, -2)",
    #         #"Ref($close, -1)",
    #         #"Ref($close, -2)/Ref($close, -1) - 1",
    #     ],
    #     start_time="2025-04-24",
    #     end_time="2025-05-18",
    # )

    handler = Alpha(
        instruments="all",
        start_time="2015-01-01",
        end_time="2025-05-22",
        # learn_processors=[
        #     DropnaLabel(
        #         fields_group="label",
        #     ),
        #     ZScoreNorm(
        #         fit_start_time="2013-01-01",
        #         fit_end_time="2022-12-31",
        #         fields_group="feature",
        #     ),
        # ],
        # fit_start_time="2018-09-10",
        # fit_end_time="2018-10-31",
    )

    df = handler.fetch(col_set=["feature", "label"])

    #df.to_csv("c.csv")

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
                "handler": {
                    "class": "Alpha",
                    "module_path": "test",
                    "kwargs": {
                        "instruments": "all",
                        "start_time": "2013-01-01",
                        "end_time": "2025-05-22",
                    },
                },
                "segments": {
                    "train": ("2013-01-01", "2020-12-31"),
                    "valid": ("2021-01-01", "2022-12-31"),
                    "test": ("2023-01-01", "2025-05-22"),
                },
            },
        },
        "record": {
            "class": "SignalRecord",
            "module_path": "qlib.workflow.record_temp",
            "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"},
        },
    }

    # task_train(config, experiment_name="future_alpha_test")

    # with R.start():

    #     dataset = init_instance_by_config(config["dataset"])
    #     model = init_instance_by_config(config["model"])

    #     model.fit(dataset)

    #     record = init_instance_by_config(
    #         {
    #             "class": "SignalRecord",
    #             "module_path": "qlib.workflow.record_temp",
    #             "kwargs": {"model": model, "dataset": dataset},
    #         }
    #     )

    #     record.generate()
