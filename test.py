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
col1 = 2
col2 = 3

X, y = cnS2[:, col1], cnS2[:, col2]  # 特征矩阵 X 和目标变量 y
X = X.reshape(-1, 1)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)






