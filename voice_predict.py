'''
男女声音分类
'''
import numpy as np
import pandas as pd

# 数据加载
df = pd.read_csv('voice.csv')
# print(df.head())

# corr相关系数函数
# print(df.corr())
# 判断缺失值
print(df.isnull().sum())
print(df.shape)

print('样本个数:{}'.format(df.shape[0]))
print('男性样本:{}'.format(df[df.label == 'male'].shape[0]))
print('女性样本:{}'.format(df[df.label == 'female'].shape[0]))

# 分割features和label
X = df.iloc[:,:-1]
y = df.iloc[:, -1]
print(X.head())
print(y.head())

# label encoder
from sklearn.preprocessing import LabelEncoder
gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)

# 数据规范化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)

# 数据切分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 使用SVM进行训练和预测
from sklearn.svm import SVC
from sklearn import metrics
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print('SVM 预测结果:', y_pred)
print('SVM 准确率:', metrics.accuracy_score(y_test,y_pred))

# 使用XGBoost进行训练和预测
import xgboost as xgb
param = {'boosting_type':'gbdt',
                         'objective' : 'binary:logistic', #
                         'eval_metric' : 'auc',
                         'eta' : 0.01,
                         'max_depth' : 15,
                         'colsample_bytree':0.8,
                         'subsample': 0.9,
                         'subsample_freq': 8,
                         'alpha': 0.6,
                         'lambda': 0,
        }
train_data = xgb.DMatrix(X_train, label=y_train)
valid_data = xgb.DMatrix(X_test, label=y_test)
test_data = xgb.DMatrix(X_test)

model = xgb.train(param, train_data, evals=[(train_data, 'train'), (valid_data, 'valid')], num_boost_round = 10000, early_stopping_rounds=200, verbose_eval=25)
y_pred = model.predict(test_data)
print('XGBoost 预测结果:', y_pred)