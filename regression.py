import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def linear_regression(X, y):
    """
    该函数可以用于线性回归，并计算RMSE、MAE、R方等的误差
    :param X: 传入自变量数据
    :param y: 传入因变量数据
    :return: 总共返回五个值，分别是线性回归模型的RMSE、MAE、R方、截距和斜率
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred = linear_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    beta_0 = linear_model.intercept_      # 计算模型的截距
    beta_1 = linear_model.coef_[0]        # 计算模型的斜率

    return rmse, mae, r2, beta_0, beta_1


def ridge_regression(X, y):
    """
    该函数可以用于岭回归，并计算RMSE、MAE、R方等的误差
    :param X: 传入自变量数据
    :param y: 传入因变量数据
    :return: 总共返回三个值，分别是岭回归模型的RMSE、MAE、R方
    """
    # 建立岭回归模型
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    linear_model = Ridge(alpha=1.0)  # 在这里设置alpha参数
    linear_model.fit(X_train, y_train)
    y_pred = linear_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return rmse, mae, r2

    # 回归误差图
    # # 创建散点图
    # plt.figure(figsize=(8, 8))  # 将图制作成方形
    # plt.scatter(y_test, y_pred, alpha=0.8, color='red')  # 提高点的亮度和改变 颜色
    # # plt.scatter(y, y_pred, alpha=0.8, color='red')  # 提高点的亮度和改变颜色
    # # plt.scatter(y_train, y_train_pred, alpha=0.8, color='blue')
    # plt.xlabel("实际值", fontsize=14)  # 增加坐标轴标签的字体大小
    # plt.ylabel("预测值", fontsize=14)
    #
    # # 在图上添加R²和RMSE、M的值
    # plt.text(plt.xlim()[0] + 300, plt.ylim()[1] - 100, f'R方: {r2:.5f}', fontsize=12, color='blue')
    # plt.text(plt.xlim()[0] + 300, plt.ylim()[1] - 500, f'MSE: {rmse:.3f}', fontsize=12, color='red')
    # plt.text(plt.xlim()[0] + 300, plt.ylim()[1] - 900, f'MSE: {mae:.3f}', fontsize=12, color='green')
    #
    # plt.plot(y, y, color='gray', linestyle='--', linewidth=2)  # 对角线，表示理想情况
    #
    # # 图的标题
    # plt.title('')
    # # 保存图
    # plt.savefig('1.pdf')
    # # 显示图
    # plt.show()

