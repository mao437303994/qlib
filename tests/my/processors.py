import pandas as pd
from qlib.data.dataset.processor import Processor


class Clip(Processor):
    def __init__(self, col_list=[], clip_outlier=None):
        self.col_list = col_list
        self.clip_outlier = clip_outlier

    def __call__(self, df):
        if self.clip_outlier is not None:
            if isinstance(df.columns, pd.MultiIndex):
                columns = df.columns[
                    df.columns.get_level_values(-1).isin(self.col_list)
                ]
            else:
                columns = df.columns[df.columns.isin(self.col_list)]

            mask = pd.DataFrame(True, index=df.index, columns=columns)

            ul, ll = self.clip_outlier
            if ul is not None:
                mask = mask & (df[columns] < ul)
            if ll is not None:
                mask = mask & (df[columns] > ll)

            row_mask = mask.all(axis=1)
            df = df[row_mask]
        return df

    def readonly(self):
        return True
