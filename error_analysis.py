import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from regression import ridge_regression, linear_regression

data = loadmat('matlab.mat')
# cnS2 数据格式为 76*4 （76行数据，4列，分别是土壤有效态，土壤，根系和水稻中的cd含量）
cnS2 = data['cnS2']
excel_file_path = '！湘潭数据_cn.xlsx'
df = pd.read_excel(excel_file_path, sheet_name=2)
cd_type = ['土壤有效态Cd', '土壤中Cd', '根系中Cd', '水稻中Cd']
error_type = ['rmse', 'mae', 'r2']
group_num = 11     # 设置分组数量

file_name = 'result.xlsx'
writer = pd.ExcelWriter(file_name, engine='xlsxwriter')

for j in range(17, 28):
    if j == 24:
        continue

    factor_name = df.columns[j].strip(' [ mgkg ]')    # 获取对应列列名

    # 排除带有字符串（数据异常）的行
    rows_to_exclude = df[df.iloc[:, j].apply(lambda x: isinstance(x, str))].index
    df = df.drop(rows_to_exclude)
    cnS2 = np.delete(cnS2, rows_to_exclude, axis=0)

    tmpkk = df.iloc[:, j]       # 获取影响因素对应的列

    # th是把用于筛选的特征分成100分，这里太多了（就76个数），其实分成10份左右就好
    min_value = np.min(tmpkk)
    max_value = np.max(tmpkk)
    step = (max_value - min_value) / (group_num + 1)
    th = np.linspace(min_value + step, max_value - step, num=group_num)

    # for col in range(3):
    # 这里col2 = 3 是 农作物中的的cd，col1 = 0,1,2，分别是土壤有效态，土壤，和根系中的cd，改变col1 的值就可以改变比对
    col1 = 2
    col2 = 3

    # 记录全局误差，用于作为基线
    X, y = cnS2[:, col1], cnS2[:, col2]  # 特征矩阵 X 和目标变量 y
    X = X.reshape(-1, 1)
    total = linear_regression(X, y)[:3]
    # print(total)
    intercept = linear_regression(X, y)[3]    # 线性模型的截距
    slope = linear_regression(X, y)[4]        # 线性模型的斜率
    # print(intercept, slope)

    # 这里的101是我把数据分成了100+1 份，如果是10分就改成11，以此类比
    co = np.zeros((group_num, 8))

    remove = []             # 创建空列表，记录不能进行岭回归的分组序号

    for i in range(len(th)):
        k = th[i]
        # 分别筛选小于等于和大于阈值的数
        idx1 = np.where(tmpkk <= k)[0]
        # print(f'分组{i}')
        # print(idx1)
        idx2 = np.where(tmpkk > k)[0]
        # print(idx2)

        # 计算分开两组的各自数量
        co[i, 0] = len(idx1)
        co[i, 1] = len(idx2)
        print()
        # 计算correlation
        try:
            co[i, 2], co[i, 4], co[i, 6] = linear_regression(cnS2[idx1, col1].reshape(-1, 1), cnS2[idx1, col2])[:3]
            co[i, 3], co[i, 5], co[i, 7] = linear_regression(cnS2[idx2, col1].reshape(-1, 1), cnS2[idx2, col2])[:3]
        except ValueError:
            remove.append(i)
            continue

    # 如果找不到强行计算误差，可能会出现nan，把它赋值为0
    co[np.isnan(co)] = 0
    co = np.around(co, decimals=4)

    # filtered_th = th[~np.isin(np.arange(len(th)), remove)]  # 删除th中报错的分组
    # print(len(filtered_th))
    # filtered_co = np.delete(co, remove, axis=0)             # 删除co数组中报错的行
    # print(filtered_co.shape[0])
    filtered_th = th
    filtered_co = co
    # 可视化设置
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei或其他支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

    # 建立新的dataframe数据帧用于记录结果，最终输出到excel表格中
    row_names = []
    for i in range(11):
        row_names.append(f'{round(th[i], 4)}')
    result = pd.DataFrame(co, index=row_names, columns=[
        '分组1样本数量', '分组2样本数量', '分组1RMSE', '分组2RMSE', '分组1MAE', '分组2MAE', '分组1R方', '分组2R方'])

    for m in range(3):
        m1 = 2 * (m + 1)
        m2 = m1 + 1
        # 合成误差
        aver_error = (filtered_co[:, m1] + filtered_co[:, m2]) / 2     # 两组误差取平均值

        # 配平合成误差
        trim_error = (filtered_co[:, m1] * filtered_co[:, 0] + filtered_co[:, m2] * filtered_co[:, 1]) / \
                     (filtered_co[:, 0] + filtered_co[:, 1])      # 配平方式为两组误差分别乘以各组数量的占比

        # 整体误差
        total_error = np.ones(11 - len(remove)) * total[m]

        # 填充结果矩阵
        result[f'合成{error_type[m]}'] = aver_error.round(4)
        result[f'合成{error_type[m]}占比'] = np.vectorize(lambda value: f'{value:.2f}%')(aver_error / total[m] * 100)
        result[f'配平合成{error_type[m]}'] = trim_error.round(4)
        result[f'配平合成{error_type[m]}占比'] = np.vectorize(lambda value: f'{value:.2f}%')(trim_error / total[m] * 100)

        # 输出最值
        # print(f'{factor_name}')
        # print(f'合成{error_type[m]}的最小值为{np.min(aver_error):.4f}，此时的{factor_name}为{th[np.argmin(aver_error)]:.3f}')
        # print(f'配平合成{error_type[m]}的最小值为{np.min(trim_error):.4f}，此时的{factor_name}为{th[np.argmin(trim_error)]:.3f}')
        # print(f'整体{error_type[m]}的值为{total[m]:.4f}')

        # # 可视化
        # plt.figure(figsize=(10, 8))
        # plt.plot(filtered_th, aver_error)
        # plt.plot(filtered_th, trim_error)
        # plt.plot(filtered_th, total_error)
        # plt.legend([f'合成{error_type[m]}', f'配平合成{error_type[m]}', f'整体{error_type[m]}'])
        # plt.xlabel(f"{factor_name}浓度")
        # plt.ylabel(f"{error_type[m]}")
        # plt.title(f'{factor_name}对{cd_type[col1]}和{cd_type[col2]}的{error_type[m]}的影响程度')
        # # plt.show()          # 显示图
        # if m == 0 or m == 1:
        #     plt.savefig(f"error_figures/{factor_name}对{cd_type[col1]}和{cd_type[col2]}的{error_type[m]}的影响程度.png", format='png')

    result.iloc[remove, 2:] = '-'  # 将异常的数据标记出来
    new_order = ['分组1样本数量', '分组2样本数量', '分组1RMSE', '分组2RMSE', '合成rmse', '配平合成rmse', '合成rmse占比',
                 '配平合成rmse占比', '分组1MAE', '分组2MAE', '合成mae', '配平合成mae', '合成mae占比', '配平合成mae占比',
                 '分组1R方', '分组2R方', '合成r2', '配平合成r2', '合成r2占比', '配平合成r2占比']
    result = result[new_order]
    result.to_excel(writer, sheet_name=f'{factor_name}')
    # writer.save()
    # worksheet = writer[f'{factor_name}']
    # worksheet.cell(row=1, column=1, value='分组阈值')
#
writer.save()
