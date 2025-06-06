from qlib.contrib.data.handler import Alpha158, Alpha360


class Alpha(Alpha158):
    def get_feature_config(self):
        fields, names = super().get_feature_config()

        # # 1. 动量与收益率因子
        # fields += [
        #     "($close/Ref($close,1))-1",  # 昨日涨跌幅
        #     "($close/Ref($close,5))-1",  # 5日动量
        #     "($close/Ref($close,20))-1",  # 20日动量
        #     "($close-$open)/$open",  # 当日振幅
        #     "($close-$low)/($high-$low+1e-6)",  # 收盘在区间位置
        # ]
        # names += ["RET1", "RET5", "RET20", "INTRADAY_RET", "CLOSE_POS"]

        # # 2. 均值回归与偏离
        # fields += [
        #     "($close-EMA($close,5))/EMA($close,5)",  # 5日均线偏离
        #     "($close-EMA($close,20))/EMA($close,20)",  # 20日均线偏离
        #     "($close-Min($close,20))/Min($close,20)",  # 20日最低价偏离
        #     "($close-Max($close,20))/Max($close,20)",  # 20日最高价偏离
        # ]
        # names += ["BIAS5", "BIAS20", "MIN_BIAS20", "MAX_BIAS20"]

        # # 3. 波动率因子
        # fields += [
        #     "Std($close/Ref($close,1)-1,5)",  # 5日收益率波动率
        #     "Std($close/Ref($close,1)-1,20)",  # 20日收益率波动率
        #     "($high-$low)/Ref($close,1)",  # 当日振幅
        # ]
        # names += ["STD5", "STD20", "RANGE"]

        # # 4. 成交量与持仓量因子
        # fields += [
        #     "($volume/EMA($volume,5))-1",  # 5日成交量变化
        #     "($oi/EMA($oi,5))-1",  # 5日持仓量变化
        #     "($volume-Ref($volume,1))/Ref($volume,1)",  # 成交量日变化
        #     "($oi-Ref($oi,1))/Ref($oi,1)",  # 持仓量日变化
        # ]
        # names += ["VOL_CHG5", "OI_CHG5", "VOL_CHG1", "OI_CHG1"]

        # 5. 周期因子
        fields += ["$month/100", "$day/100"]
        names += ["MONTH", "DAY"]

        return fields, names

    def get_label_config(self):
        return ["(Ref($close, -1)/$close - 1) * 100"], ["LABEL0"]
