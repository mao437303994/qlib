from qlib.contrib.data.handler import Alpha158, Alpha360


class Alpha(Alpha158):
    def get_feature_config(self):
        fields, names = super().get_feature_config()
        # fields, names = ([], [])
        # # 添加自定义特征

        fields += [
            "$close",  # 收盘价
            "$open",  # 开盘价
            "$high",  # 最高价
            "$low",  # 最低价
            "($close/Ref($close,1))-1",  # 昨日涨跌幅
            "($close/Ref($close,5))-1",  # 5日动量
            "($close/Ref($close,20))-1",  # 20日动量
            "($close-$open)/$open",  # 当日振幅
            "($close-$low)/($high-$low+1e-6)",  # 收盘在区间位置
            "($close-EMA($close,5))/EMA($close,5)",  # 5日均线偏离
            "($close-EMA($close,20))/EMA($close,20)",  # 20日均线偏离
            "($close-Min($close,20))/Min($close,20)",  # 20日最低价偏离
            "($close-Max($close,20))/Max($close,20)",  # 20日最高价偏离
            "Std(($close/Ref($close,1)-1),5)",  # 5日收益率波动率
            "Std(($close/Ref($close,1)-1),20)",  # 20日收益率波动率
            "($high-$low)/Ref($close,1)",  # 当日振幅
            "($volume/EMA($volume,5))-1",  # 5日成交量变化
            "($oi/EMA($oi,5))-1",  # 5日持仓量变化
            "($volume-Ref($volume,1))/Ref($volume,1)",  # 成交量日变化
            "($oi-Ref($oi,1))/Ref($oi,1)",  # 持仓量日变化
        ]

        names += [
            "CLOSE",  # 收盘价
            "OPEN",  # 开盘价
            "HIGH",  # 最高价
            "LOW",  # 最低价
            "RET1",  # 昨日涨跌幅
            "RET5",  # 5日动量
            "RET20",  # 20日动量
            "INTRADAY_RET",  # 当日振幅
            "CLOSE_POS",  # 收盘在区间位置
            "BIAS5",  # 5日均线偏离
            "BIAS20",  # 20日均线偏离
            "MIN_BIAS20",  # 20日最低价偏离
            "MAX_BIAS20",  # 20日最高价偏离
            "MSTD5",  # 5日收益率波动率
            "MSTD20",  # 20日收益率波动率
            "RANGE",  # 当日振幅
            "VOL_CHG5",  # 5日成交量变化
            "OI_CHG5",  # 5日持仓量变化
            "VOL_CHG1",  # 成交量日变化
            "OI_CHG1",  # 持仓量日变化
        ]

        fields += [
            "Month($timestamp)",
            "Day($timestamp)",
            "Sin(2*PI*Month($timestamp)/12)",
            "Cos(2*PI*Month($timestamp)/12)",
            "Sin(2*PI*Day($timestamp)/31)",
            "Cos(2*PI*Day($timestamp)/31)",
        ]

        names += [
            "MONTH",
            "DAY",
            "MONTH_SIN",
            "MONTH_COS",
            "DAY_SIN",
            "DAY_COS",
        ]

        # print(fields)
        return fields, names

    def get_label_config(self):
        return ["If((Ref($close,-1)/$close-1)>0,1,0)"], ["LABEL0"]
        #return ["(Ref($close,-1)/$close-1)*100"], ["LABEL0"]


