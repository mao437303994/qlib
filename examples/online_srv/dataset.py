from qlib.data.dataset import DatasetH
import pandas as pd


class DatasetS(DatasetH):

    def __init__(self, handler, data_ratio=[0.6, 0.2, 0.2], **kwargs):
        assert pd.Series(data_ratio).sum() == 1, "data_ratio must sum to 1"
        super().__init__(handler=handler, segments={}, **kwargs)
        df = self.handler.fetch(col_set=["label"])
        self.segments = self._get_time_segments(
            df,
            train_ratio=data_ratio[0],
            valid_ratio=data_ratio[1],
            test_ratio=data_ratio[2],
        )
        print(f"Dataset segments: {self.segments}")

    def _get_time_segments(self, df, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2):
        all_dates = sorted(df.index.get_level_values("datetime").unique())
        n = len(all_dates)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        n_test = n - n_train - n_valid

        train_dates = all_dates[:n_train]
        valid_dates = all_dates[n_train : n_train + n_valid]
        test_dates = all_dates[n_train + n_valid :]

        return {
            "train": (str(train_dates[0]), str(train_dates[-1])),
            "valid": (str(valid_dates[0]), str(valid_dates[-1])),
            "test": (str(test_dates[0]), str(test_dates[-1])),
        }
