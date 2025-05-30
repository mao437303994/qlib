from qlib.contrib.data.handler import Alpha158


class Alpha(Alpha158):
    def get_feature_config(self):
        f, l = super().get_feature_config()
        f += ["$oi", "$oi/Ref($oi,1)-1"]
        l += ["OI", "OI_CHG"]

        f += ["$month", "$day"]
        l += ["MONTH", "DAY"]

        return f, l
