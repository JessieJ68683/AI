import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor
from sklearn.preprocessing import PolynomialFeatures,MinMaxScaler,StandardScaler
from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score,cross_val_predict,KFold
from xgboost import XGBRegressor

#导入数据
with open("./zhengqi_train.txt")  as fr:
    data_train=pd.read_table(fr,sep="\t")
with open("./zhengqi_test.txt") as fr_test:
    data_test=pd.read_table(fr_test,sep="\t")

data_train.info()
data_test.info()
data_train.head()

#核密度图
rows = len(data_test.columns)
cols = 6
i = 1
plt.figure(figsize=(4 * cols, 4 * rows))
for column in data_test.columns:
    ax = plt.subplot(rows, cols, i)
    ax = sns.kdeplot(data_train[column], color="Red", shade=True)
    ax = sns.kdeplot(data_test[column], color="Blue", shade=True)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax = ax.legend(["train", "test"])
    i += 1
plt.show()

#放大查看分布不一致的特征量
cols = 3
rows = 2
plt.figure(figsize=(5 * cols, 5 * rows))
for i, column in enumerate(["V5", "V9", "V11", "V17", "V22", "V28"]):
    ax = plt.subplot(rows, cols, i + 1)
    ax = sns.kdeplot(data_train[column], color="Red", shade=True)
    ax = sns.kdeplot(data_test[column], color="Blue", shade=True)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax = ax.legend(["train", "test"])
plt.show()

#将训练数据与测试数据合并
data_test["oringin"]="test"
data_train["oringin"]="train"
steam_data=pd.concat([data_train, data_test], axis=0, ignore_index=True)

#删除分布不一致的特征
steam_data.drop(["V5", "V9", "V11", "V17", "V22", "V28"], axis=1, inplace=True)

#合并后的数据最大最小归一化
num_cols=list(steam_data.columns)
num_cols.remove("oringin")
def data_normalize(column):
    return (column-column.min())/(column.max()-column.min())
normalize_cols = [col for col in num_cols if col != 'target']
steam_data[normalize_cols] = steam_data[normalize_cols].apply(data_normalize, axis=0)
#display(steam_data.describe())

#展示Box-Cox进行变换前后的特征分布
after_cols = 6
after_rows = len(num_cols) - 1
plt.figure(figsize=(4 * after_cols, 4 * after_rows))
i = 0

for vargue in num_cols:
    if vargue != 'target':
        drop_data = steam_data[[vargue, 'target']].dropna()
        i += 1
        plt.subplot(after_rows, after_cols, i)
        sns.distplot(drop_data[vargue], fit=stats.norm)
        plt.title(vargue + ' Original')
        plt.xlabel('')

        i += 1
        plt.subplot(after_rows, after_cols, i)
        _ = stats.probplot(drop_data[vargue], plot=plt)
        plt.title('skew=' + '{:.4f}'.format(stats.skew(drop_data[vargue])))
        plt.xlabel('')
        plt.ylabel('')

        i += 1
        plt.subplot(after_rows, after_cols, i)
        plt.plot(drop_data[vargue], drop_data['target'], '.', alpha=0.5)
        plt.title('corr=' + '{:.2f}'.format(np.corrcoef(drop_data[vargue], drop_data['target'])[0][1]))

        i += 1
        plt.subplot(after_rows, after_cols, i)
        vargue_transform, vargue_lam = stats.boxcox(drop_data[vargue].dropna() + 1)
        vargue_transform = data_normalize(vargue_transform)
        sns.distplot(vargue_transform, fit=stats.norm)
        plt.title(vargue + ' Tramsformed')
        plt.xlabel('')

        i += 1
        plt.subplot(after_rows, after_cols, i)
        _ = stats.probplot(vargue_transform, plot=plt)
        plt.title('skew=' + '{:.4f}'.format(stats.skew(vargue_transform)))
        plt.xlabel('')
        plt.ylabel('')

        i += 1
        plt.subplot(after_rows, after_cols, i)
        plt.plot(vargue_transform, drop_data['target'], '.', alpha=0.5)
        plt.title('corr=' + '{:.2f}'.format(np.corrcoef(vargue_transform, drop_data['target'])[0][1]))

cols_transform= steam_data.columns[0:-2]
for col in cols_transform:
    steam_data.loc[:, col], _ = stats.boxcox(steam_data.loc[:, col] + 1)

#计算分位数并展示
print(steam_data.target.describe())
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.distplot(steam_data.target.dropna(), fit=stats.norm)
plt.subplot(1,2,2)
_=stats.probplot(steam_data.target.dropna(), plot=plt)

#对数据进行对数变换，让其更符合正态分布
data_final = data_train.target
data_train.target1 =np.power(1.5, data_final)
print(data_train.target1.describe())

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.distplot(data_train.target1.dropna(),fit=stats.norm)
plt.subplot(1,2,2)
_=stats.probplot(data_train.target1.dropna(), plot=plt)

#使用简单交叉验证对模型进行验证和切分数据
def get_data_train():
    train_select = steam_data[steam_data["oringin"] == "train"]
    train_select["label"]=data_train.target1
    steam_y = train_select.target
    steam_X = train_select.drop(["oringin", "target", "label"], axis=1)
    steam_train_X, steam_valid_X, steam_train_y, steam_valid_y=train_test_split(steam_X, steam_y, test_size=0.3, random_state=100)
    return steam_train_X, steam_valid_X, steam_train_y, steam_valid_y

def get_data_test():
    test_select = steam_data[steam_data["oringin"] == "test"].reset_index(drop=True)
    return test_select.drop(["oringin","target"],axis=1)

#使用MSE和RMSE作为模型的评分函数
def steam_mse(steam_y_ture, steam_y_pred):
    return mean_squared_error(steam_y_ture, steam_y_pred)
def steam_rmse(steam_y_true, steam_y_pred):
    difference = steam_y_pred - steam_y_true
    sum_temp = sum(difference ** 2)
    m = len(steam_y_pred)
    return np.sqrt(sum_temp / m)

steam_score_rmse = make_scorer(steam_rmse, greater_is_better=False)
steam_score_mse = make_scorer(steam_mse, greater_is_better=False)

#用模型预测的方法获取异常数据
def get_abnormal_data(steam_model, steam_X, steam_y, sigma=3):
    #用模型预测y值
    try:
        steam_y_pred = pd.Series(steam_model.predict(steam_X), index=steam_y.index)
    except:
        steam_model.fit(steam_X, steam_y)
        steam_y_pred = pd.Series(steam_model.predict(steam_X), index=steam_y.index)

    #计算偏差
    steam_resid = steam_y - steam_y_pred
    steam_mean_resid = steam_resid.mean()
    steam_std_resid = steam_resid.std()

    #计算频率
    steam_z = (steam_resid - steam_mean_resid) / steam_std_resid
    abnormals = steam_z[abs(steam_z) > sigma].index

    #打印结果并画图展示
    print('R2=', steam_model.score(steam_X, steam_y))
    print('rmse=', steam_rmse(steam_y, steam_y_pred))
    print("mse=", steam_mse(steam_y, steam_y_pred))
    print('---------------------------------------')
    print('mean of residuals:', steam_mean_resid)
    print('std of residuals:', steam_std_resid)
    print('---------------------------------------')
    print(len(abnormals), 'abnormals:')
    print(abnormals.tolist())

    plt.figure(figsize=(15, 5))
    ax_131 = plt.subplot(1, 3, 1)
    plt.plot(steam_y, steam_y_pred, '.')
    plt.plot(steam_y.loc[abnormals], steam_y_pred.loc[abnormals], 'ro')
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('y')
    plt.ylabel('y_pred')

    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(steam_y, steam_y - steam_y_pred, '.')
    plt.plot(steam_y.loc[abnormals], steam_y.loc[abnormals] - steam_y_pred.loc[abnormals], 'ro')
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('y')
    plt.ylabel('y - y_pred')

    ax_133 = plt.subplot(1, 3, 3)
    steam_z.plot.hist(bins=50, ax=ax_133)
    steam_z.loc[abnormals].plot.hist(color='r', bins=50, ax=ax_133)
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('steam_z')
    plt.savefig('abnormals.png')

    return abnormals

#岭回归模型寻找异常值
steam_train_X, steam_valid_X, steam_train_y, steam_valid_y = get_data_train()
test=get_data_train()
abnormals = get_abnormal_data(Ridge(), steam_train_X, steam_train_y)

X_abnormals=steam_train_X.loc[abnormals]
y_abormals=steam_train_y.loc[abnormals]
X_final=steam_train_X.drop(abnormals)
y_final=steam_train_y.drop(abnormals)

#删除异常值函数
def get_del_abnormal_data():
    y=y_final.copy()
    X=X_final.copy()
    return X, y

#网格搜索法训练模型
def train_steam_model(steam_model, param_grid=[], steam_X=[], steam_y=[], steam_split=5, steam_repeat=5):
    if len(steam_y) == 0:
        steam_X, steam_y = get_del_abnormal_data()

    #交叉验证
    rkfold = RepeatedKFold(n_splits=steam_split, n_repeats=steam_repeat)

    if len(param_grid) > 0:
        #设置参数
        steam_gsearch = GridSearchCV(steam_model, param_grid, cv=rkfold,
                               scoring="neg_mean_squared_error",
                               verbose=1, return_train_score=True)
        #搜索
        steam_gsearch.fit(steam_X, steam_y)

        # 提取出最好的模型
        steam_model = steam_gsearch.best_estimator_
        model_best_index = steam_gsearch.best_index_

        grid_results = pd.DataFrame(steam_gsearch.cv_results_)
        model_CVmean = abs(grid_results.loc[model_best_index, 'mean_test_score'])
        model_CVstd = grid_results.loc[model_best_index, 'std_test_score']

    else:
        grid_results = []
        model_CVresults = cross_val_score(steam_model, steam_X, steam_y, scoring="neg_mean_squared_error", cv=rkfold)
        model_CVmean = abs(np.mean(model_CVresults))
        model_CVstd = np.std(model_CVresults)
    #合并
    steam_cv_score = pd.Series({'mean': model_CVmean, 'std': model_CVstd})

    #用模型预测y值
    steam_y_pred = steam_model.predict(steam_X)

    #输出模型的效果
    print('----------------------')
    print(steam_model)
    print('----------------------')
    print('score=', steam_model.score(steam_X, steam_y))
    print('rmse=', steam_rmse(steam_y, steam_y_pred))
    print('mse=', steam_mse(steam_y, steam_y_pred))
    print('cross_val: mean=', model_CVmean, ', std=', model_CVstd)

    #结果画图
    steam_y_pred = pd.Series(steam_y_pred, index=steam_y.index)
    steam_resid = steam_y - steam_y_pred
    steam_mean_resid = steam_resid.mean()
    steam_std_resid = steam_resid.std()
    steam_z = (steam_resid - steam_mean_resid) / steam_std_resid
    steam_abnormals = sum(abs(steam_z) > 3)

    plt.figure(figsize=(15, 5))
    ax_131 = plt.subplot(1, 3, 1)
    plt.plot(steam_y, steam_y_pred, '.')
    plt.xlabel('y')
    plt.ylabel('steam_y_pred')
    plt.title('corr = {:.3f}'.format(np.corrcoef(steam_y, steam_y_pred)[0][1]))
    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(steam_y, steam_y - steam_y_pred, '.')
    plt.xlabel('y')
    plt.ylabel('y - steam_y_pred')
    plt.title('std steam_resid = {:.3f}'.format(steam_std_resid))

    ax_133 = plt.subplot(1, 3, 3)
    steam_z.plot.hist(bins=50, ax=ax_133)
    plt.xlabel('steam_z')
    plt.title('{:.0f} samples with steam_z>3'.format(steam_abnormals))

    return steam_model, steam_cv_score, grid_results

steam_opt_models = dict()
steam_models_score = pd.DataFrame(columns=['mean', 'std'])
#5折交叉验证
steam_split=5
steam_repeat=5

#单一模型进行训练预测
#1、K近邻
model = 'KNeighbors'
steam_opt_models[model] = KNeighborsRegressor()
param_grid = {'n_neighbors':np.arange(3,11,1)}

steam_opt_models[model], steam_cv_score, grid_results = train_steam_model(steam_opt_models[model], param_grid=param_grid,
                                                                          steam_split=steam_split, steam_repeat=1)

steam_cv_score.name = model
steam_models_score = steam_models_score.append(steam_cv_score)

#画图展示参数与模型评价指标的误差棒图
plt.figure()
plt.errorbar(np.arange(3,11,1), abs(grid_results['mean_test_score']), abs(grid_results['std_test_score']) / np.sqrt(steam_split * 1))
plt.xlabel('n_neighbors')
plt.ylabel('score')

#2、岭回归
model = 'Ridge'
steam_opt_models[model] = Ridge()
steam_alpha_range = np.arange(0.25, 6, 0.25)
param_grid = {'alpha': steam_alpha_range}

steam_opt_models[model], steam_cv_score, grid_results = train_steam_model(steam_opt_models[model], param_grid=param_grid,
                                                                          steam_split=steam_split, steam_repeat=steam_repeat)

steam_cv_score.name = model
steam_models_score = steam_models_score.append(steam_cv_score)

#画图展示参数与模型评价指标的误差棒图
plt.figure()
plt.errorbar(steam_alpha_range, abs(grid_results['mean_test_score']),
             abs(grid_results['std_test_score']) / np.sqrt(steam_split * steam_repeat))
plt.xlabel('alpha')
plt.ylabel('score')

#3、Lasso回归
model = 'Lasso'
steam_opt_models[model] = Lasso()
steam_alpha_range = np.arange(1e-4, 1e-3, 4e-5)
param_grid = {'alpha': steam_alpha_range}

steam_opt_models[model], steam_cv_score, grid_results = train_steam_model(steam_opt_models[model], param_grid=param_grid,
                                                                          steam_split=steam_split, steam_repeat=steam_repeat)

steam_cv_score.name = model
steam_models_score = steam_models_score.append(steam_cv_score)

#画图展示参数与模型评价指标的误差棒图
plt.figure()
plt.errorbar(steam_alpha_range, abs(grid_results['mean_test_score']), abs(grid_results['std_test_score']) / np.sqrt(steam_split * steam_repeat))
plt.xlabel('alpha')
plt.ylabel('score')

#4、ElasticNet回归
model ='ElasticNet'
steam_opt_models[model] = ElasticNet()
steam_alpha_range = np.arange(1e-4, 1e-3, 1e-4)
param_grid = {'alpha': steam_alpha_range,
              'l1_ratio': np.arange(0.1,1.0,0.1),
              #'l1_ratio': [0.6],
              'max_iter':[100000]}

steam_opt_models[model], steam_cv_score, grid_results = train_steam_model(steam_opt_models[model], param_grid=param_grid,
                                                                          steam_split=steam_split, steam_repeat=1)

steam_cv_score.name = model
steam_models_score = steam_models_score.append(steam_cv_score)

#画图展示参数与模型评价指标的误差棒图
plt.figure()
plt.errorbar(steam_alpha_range, abs(grid_results['mean_test_score']), abs(grid_results['std_test_score']) / np.sqrt(steam_split * steam_repeat))
plt.xlabel('alpha')
plt.ylabel('score')

#5、SVR回归
model='LinearSVR'
steam_opt_models[model] = LinearSVR()

crange = np.arange(0.1,1.0,0.1)
param_grid = {'C':crange,
             'max_iter':[1000]}

steam_opt_models[model], steam_cv_score, grid_results = train_steam_model(steam_opt_models[model], param_grid=param_grid,
                                                                          steam_split=steam_split, steam_repeat=steam_repeat)

steam_cv_score.name = model
steam_models_score = steam_models_score.append(steam_cv_score)

#画图展示参数与模型评价指标的误差棒图
plt.figure()
plt.errorbar(crange, abs(grid_results['mean_test_score']), abs(grid_results['std_test_score']) / np.sqrt(steam_split * steam_repeat))
plt.xlabel('C')
plt.ylabel('score')


#模型融合方法进行训练预测
#1、GBDT模型
model = 'GradientBoosting'
steam_opt_models[model] = GradientBoostingRegressor()

param_grid = {'n_estimators':[150,250,350],
              'max_depth':[1,2,3],
              'min_samples_split':[5,6,7]}

steam_opt_models[model], steam_cv_score, grid_results = train_steam_model(steam_opt_models[model], param_grid=param_grid,
                                                                          steam_split=steam_split, steam_repeat=1)

steam_cv_score.name = model
steam_models_score = steam_models_score.append(steam_cv_score)

#2、XGB模型
model = 'XGB'
steam_opt_models[model] = XGBRegressor()

param_grid = {'n_estimators':[100,200,300,400,500],
              'max_depth':[1,2,3],
             }

steam_opt_models[model], steam_cv_score, grid_results = train_steam_model(steam_opt_models[model], param_grid=param_grid,
                                                                          steam_split=steam_split, steam_repeat=1)

steam_cv_score.name = model
steam_models_score = steam_models_score.append(steam_cv_score)

#3、随机森林模型
model = 'RandomForest'
steam_opt_models[model] = RandomForestRegressor()

param_grid = {'n_estimators':[100,150,200],
              'max_features':[8,12,16,20,24],
              'min_samples_split':[2,4,6]}

steam_opt_models[model], steam_cv_score, grid_results = train_steam_model(steam_opt_models[model], param_grid=param_grid,
                                                                          steam_split=5, steam_repeat=1)

steam_cv_score.name = model
steam_models_score = steam_models_score.append(steam_cv_score)







