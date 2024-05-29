import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = loadmat('matlab.mat')
# cnS2 数据格式为 76*4 （76行数据，4列，分别是土壤有效态，土壤，根系和水稻中的cd含量）
cnS2 = data['cnS2']
excel_file_path = '！湘潭数据_cn.xlsx'
df = pd.read_excel(excel_file_path, sheet_name=2)
cd_type = ['土壤有效态Cd', '土壤中Cd', '根系中Cd', '水稻中Cd']


# 通过把tmpkk赋值不同的理化性质，用来选择作为控制的特征，例如铁或者阳离子

for j in range(17, 28):
    factor_name = df.columns[j].strip(' [ mgkg ]')    # 获取对应列列名

    # 排除带有字符串的行
    rows_to_exclude = df[df.iloc[:, j].apply(lambda x: isinstance(x, str))].index
    df = df.drop(rows_to_exclude)
    cnS2 = np.delete(cnS2, rows_to_exclude, axis=0)
    tmpkk = df.iloc[:, j]       # 获取影响相关性的列

    # th是把用于筛选的特征分成100分，这里太多了（就76个数），其实分成10份左右就好
    th = np.linspace(np.min(tmpkk), np.max(tmpkk), num=11)

    for col in range(3):
        # 这里col2 = 3 是 农作物中的的cd，col1 = 0,1,2，分别是土壤有效态，土壤，和根系中的cd，改变col1 的值就可以改变比对
        col1 = col
        col2 = 3

        # cc是全局的correlation，用于作为基线
        cc = np.corrcoef(cnS2[:, col1], cnS2[:, col2])[0, 1]


        # 这里的101是我把数据分成了100+1 份，如果是10分就改成11，以此类比
        co = np.zeros((11, 4))

        for i in range(len(th)):
            k = th[i]
            # 分别筛选大于和小于阈值的数
            idx1 = np.where(tmpkk <= k)[0]
            idx2 = np.where(tmpkk > k)[0]

            # 计算correlation
            co[i, 0] = np.corrcoef(cnS2[idx1, col1], cnS2[idx1, col2])[0, 1]
            co[i, 1] = np.corrcoef(cnS2[idx2, col1], cnS2[idx2, col2])[0, 1]
            # 计算分开两组的各自数量，用于归一化
            co[i, 2] = len(idx1)
            co[i, 3] = len(idx2)

        # 如果找不到强行计算correlation，会出现nan，把它赋值为0
        co[np.isnan(co)] = 0

        # 可视化设置
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei或其他支持中文的字体
        plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号
        plt.figure(figsize=(10, 8))

        # 画不做归一化的图
        plt.plot(np.arange(1, 12), (co[:, 0] + co[:, 1]))

        # 归一化，分别是线性归一化和 平方和归一化
        tmp = co[:, 0] * (co[:, 2] / np.sqrt(co[:, 2] ** 2 + co[:, 3] ** 2)) + co[:, 1] * (co[:, 3] / np.sqrt(co[:, 2] ** 2 + co[:, 3] ** 2))

        # plt.plot(np.arange(1, 12), co[:, 0] * tmp)       # 3种写法
        # plt.plot(np.arange(1, 12), co[:, 1] * tmp)
        plt.plot(np.arange(1, 12), (co[:, 0] + co[:, 1]) * tmp)
        # plt.plot(np.arange(1, 12), tmp)
        plt.plot(np.arange(1, 12), np.ones(11) * cc)


        plt.legend(['合成相关性', '配平合成相关性', '整体相关性'])
        plt.title(f'{factor_name}对{cd_type[col1]}和{cd_type[col2]}相关性的影响程度')
        # plt.show()          # 显示图
        # plt.savefig(f"figures/{factor_name}对{cd_type[col1]}和{cd_type[col2]}相关性的影响程度.pdf", format='pdf')



# 下面是岭回归部分
X, y = cnS2[:, col1], cnS2[:, col2]  # 特征矩阵 X 和目标变量 y

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
# 创建一个岭回归模型
ridge_model = Ridge(alpha=0.3)

# 在训练数据上拟合模型
ridge_model.fit(X_train, y_train)

# 使用模型进行预测
y_pred = ridge_model.predict(X_test)

# 计算模型的均方误差（MSE）
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MSE:", mse)
print('MAE:', mae)
print('R方:', r2)

# 查看模型的系数
coefficients = ridge_model.coef_
print("模型系数", coefficients)


