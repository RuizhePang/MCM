import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 读取数据
data = pd.read_csv('../data/summerOly_medal_counts.csv')

# 将数据透视化，设置每个国家每一年的奖牌数
pivot_data = data.pivot(index='NOC', columns='Year', values='Total')

# 设置特定年份
years = [1896, 1900, 1904, 1908, 1912, 1920, 1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020]
data = pivot_data

pivot_data = pivot_data[years]

# 进行国家循环，单独为每个国家训练模型
for country in pivot_data.index:
    if country != 'China':
        continue
    print(f"\n训练模型 - 国家: {country}")

    # 获取该国家的奖牌数据
    country_data = pivot_data.loc[country]

    # 确保数据足够进行时间序列建模
    if len(country_data) < 10:
        print(f"{country} - 数据不足以训练模型")
        continue

    # 将年份作为时间索引
    country_data.index = pd.to_datetime(country_data.index, format='%Y')

    country_data = country_data.asfreq('4YS')  # 设置频率为每年
    country_data = country_data.ffill()
    country_data = country_data.sort_index()

    model = ARIMA(country_data, order=(1, 1, 1))  # 示例参数
    model_fit = model.fit()  # 训练模型

    forecast = model_fit.forecast(steps=1)  # 预测未来1年的奖牌数

    print(f"{country} - Actual total medal numbers: ", data.loc[country,2024])

    print(f"{country} - 预测的未来1年奖牌数：", forecast)

    input_data = country_data[-10:]  # 选择最近5年数据进行预测

    model = ARIMA(input_data, order=(1, 1, 1))
    model_fit = model.fit()  # 重新训练模型

    # 预测2024年奖牌数
    prediction = model_fit.forecast(steps=1)
    print(f"{country} - 预测奖牌数：", prediction)
