import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import pandas as pd

    # 读取生成的 CSV 文件
    df = pd.read_csv("feature.csv",index_col=[0,1],header=[0,1])

    # 绘制相关性热图
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    input("Press Enter to exit...")