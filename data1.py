import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import matplotlib
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.family'] = 'SimHei'  # 'SimHei' 是一种常用的中文黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据加载
data = pd.read_excel("C:/Users/86158/Desktop/Ma-Mi-P/data1.xlsx")  # 替换为你的文件路径

#对非数值型数据进行OneHot编码
encoder = OneHotEncoder()
transformed_data = encoder.fit_transform(data[['售后服务质量']]).toarray()
transformed_df = pd.DataFrame(transformed_data, columns=encoder.get_feature_names_out(['售后服务质量']))
data = pd.concat([data.drop('售后服务质量', axis=1), transformed_df], axis=1)

# 选择特征和目标变量
X = data.drop(['销量'], axis=1)  # 假设销量列名为'销量'，其他为特征
y = data['销量']

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 决策树模型
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)
dt_predictions = dt_regressor.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_predictions)
print('Decision Tree MSE:', dt_mse)

# 随机森林模型
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
rf_predictions = rf_regressor.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
print('Random Forest MSE:', rf_mse)

# 特征重要性
features = pd.DataFrame()
features['Feature'] = X.columns
features['Importance'] = rf_regressor.feature_importances_
features_sorted = features.sort_values(by=['Importance'], ascending=False)
print(features_sorted)

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(features_sorted['Feature'], features_sorted['Importance'])
plt.xticks(rotation='vertical')
plt.show()
