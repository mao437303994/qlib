from qlib.contrib.data.handler import Alpha158


class Alpha(Alpha158):
    def get_feature_config(self):
        fields, names = super().get_feature_config()
        # fields += ["SAR(4,2,20)"]
        # names += ["SAR"]

        # fields += ["$oi", "$oi/Ref($oi,1)-1"]
        # names += ["OI", "OI_CHG"]

        # fields += ["$month", "$day"]
        # names += ["MONTH", "DAY"]

        return fields, names

    # def get_label_config(self):
    #     return ["Ref($close, -1)/$close - 1"], ["LABEL0"]
