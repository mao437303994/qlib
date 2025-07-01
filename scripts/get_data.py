#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import fire
from qlib.tests.data import GetData


if __name__ == "__main__":
    fire.Fire(GetData)
    # GetData().qlib_data(
    #     interval='1d',
    #     region="cn",
    #     delete_old=True,
    #     exists_skip=False,
    # )
