from qlib.contrib.data.handler import Alpha158


class Alpha158WithOI(Alpha158):
    """
    A class that extends Alpha158 to include IO functionality.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_feature_config(self):
        f, l = super().get_feature_config()
        f += ["$postion", "($postion - Ref($postion,1)) / Ref($postion,1)"]
        l += ["OI", "OI_CHG"]

        f += ["$month", "$day"]
        l += ["MONTH", "DAY"]

        return f, l
