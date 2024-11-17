import time

import numpy as np , matplotlib as plt
%pylab inline
from scipy import stats
from scipy.stats import chi2_contingency
from matplotlib.colors import LogNorm
from time import time

#from matplotlib import rc


import pandas as pd
%matplotlib inline
import seaborn as sns
from collections import Counter
import difflib

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import model_selection , linear_model
from sklearn.linear_model import LinearRegression , Ridge
from sklearn import preprocessing , metrics , svm, kernel_ridge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split , GridSearchCV , cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor

data_1516=pd.read_csv("cleaned_1516.csv")
data_1516.describe()
data_1516.columns

data_TCEnergy =  data_1516["SiteEnergyUse(kBtu)"]
data_EmissionsCO2= data_1516["data_EC2"]
data_1516["data_TCEnergy"] = data_TCEnergy
data_1516["data_EmissionsCO2"] = data_EmissionsCO2
sns.set_style("whitegrid")

with sns.plotting_context(context='poster'):
    sns.lmplot("data_EmissionsCO2","data_TCEnergy", data_1516, line_kws={'color': 'red'}, size=6, aspect=3)
    sns.lmplot("ENERGYSTARScore","data_EmissionsCO2", data_1516, line_kws={'color': 'red'}, size=6, aspect=3)

data_1516.columns

dt_1516= data_1516[[ 'NaturalGas(kBtu)','SiteEUI(kBtu/sf)','SteamUse(kBtu)','SourceEUI(kBtu/sf)', 'Electricity(kBtu)','ENERGYSTARScore','data_TCEnergy', 'data_EmissionsCO2']]

dt_1516.columns
corr = dt_1516.corr()
corr = corr.round(1)
plt.figure(figsize=(10, 9))
plt.title("La matrice de Corrélation entre les differents éléments")

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

sns.set(font_scale=1.2)
with sns.axes_style("white"):
    sns.heatmap(corr,  annot = True ,vmax=1, cmap="RdBu_r",square=True, mask=mask)

corr = data_1516[["data_TCEnergy","data_EmissionsCO2","ENERGYSTARScore"]].corr()
corr = corr.round(1)
plt.figure(figsize=(7, 6))
plt.title("La matrice de Corrélation entre les deux features et l'ENERGYSTARScore")

sns.set(font_scale=1.3)
with sns.axes_style("white"):
    sns.heatmap(corr,  annot = True ,vmax=1)

dumm=pd.get_dummies(data_1516.BuildingType)
merged= pd.concat([data_1516,dumm],axis='columns')
merged

dat_1516= merged.drop(['BuildingType','data_EC2'],axis='columns')
dat_1516.describe()

X_1516= dat_1516[['Campus','Multifamily HR (10+)','Multifamily LR (1-4)','Multifamily MR (5-9)','NonResidential','Nonresidential COS','Nonresidential WA','SPS-District K-12','NumberofBuildings','YearBuilt','NumberofFloors','PropertyGFATotal','Longitud','Latitud','Electricity(kBtu)','SteamUse(kBtu)','NaturalGas(kBtu)','ENERGYSTARScore','data_TCEnergy','data_EmissionsCO2','DataYear']]
X_1516.fillna(X_1516.mean(), inplace=True)
X_1516

XCTE_1516= X_1516[['Campus','Multifamily HR (10+)','Multifamily LR (1-4)','Multifamily MR (5-9)','NonResidential','Nonresidential COS','Nonresidential WA','SPS-District K-12','NumberofBuildings','YearBuilt','NumberofFloors','PropertyGFATotal','Longitud','Latitud','data_TCEnergy','DataYear']]

X= XCTE_1516.iloc[:,0:-2]
y= XCTE_1516.iloc[:,-2]
X.shape
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y,
                                test_size=0.3, random_state=38) # 30% des données dans le jeu de test

std_scale = preprocessing.StandardScaler().fit(x_train)
x_train_std = std_scale.transform(x_train)
x_test_std = std_scale.transform(x_test)

lr = linear_model.LinearRegression()
tic = time()
lr.fit(x_train_std, y_train)
print("done in {:.3f}s".format(time() - tic))

for coef in zip(x_train.columns, lr.coef_):
    print(coef)

pred_train_lr= lr.predict(x_train_std)
print(np.sqrt(mean_squared_error(y_train,pred_train_lr)))
print(r2_score(y_train, pred_train_lr))

pred_test_lr= lr.predict(x_test_std)
print( np.sqrt(mean_squared_error(y_test,pred_test_lr)))
print(r2_score(y_test, pred_test_lr))

rr = linear_model.Ridge(alpha=0.1)
tic = time()
rr.fit(x_train_std, y_train)
print("done in {:.3f}s".format(time() - tic))

for coef in zip(x_train.columns,rr.coef_):
    print(coef)

pred_train_rr= rr.predict(x_train_std)
print(np.sqrt(mean_squared_error(y_train,pred_train_rr)))
print(r2_score(y_train, pred_train_rr))

pred_test_rr= rr.predict(x_test_std)
print(np.sqrt(mean_squared_error(y_test,pred_test_rr)))
print(r2_score(y_test, pred_test_rr))

rr_cv = linear_model.Ridge()
parametr= {'alpha': [1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 30,50, 100, 200]}
ridge_r=GridSearchCV(rr_cv,parametr,scoring='neg_mean_squared_error', cv=5)
tic = time()
ridge_r.fit(x_train_std, y_train)
print("done in {:.3f}s".format(time() - tic))

for coef in zip(x_train.columns,rr.coef_):
    print(coef)
print(ridge_r.best_params_)
print(ridge_r.best_score_)

pred_train_rr= ridge_r.predict(x_train_std)
print(np.sqrt(mean_squared_error(y_train,pred_train_rr)))
print(r2_score(y_train, pred_train_rr))

pred_test_rr= ridge_r.predict(x_test_std)
print(np.sqrt(mean_squared_error(y_test,pred_test_rr)))
print(r2_score(y_test, pred_test_rr))

kr = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.01)
tic = time()
kr.fit(x_train_std, y_train)
print("done in {:.3f}s".format(time() - tic))

y_pred_test = kr.predict(x_test_std)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print ("RMSE: %.4f" % rmse)
print ("r2(score): %.4f" % r2_score(y_test, y_pred_test))

svr = GridSearchCV(SVR(kernel='rbf', gamma=0.01),
                   param_grid={"C": [1e0, 1e1, 1e2],
                               "gamma": np.logspace(-2, 2, 3)})

tic = time()
svr.fit(x_train_std, y_train)
print("done in {:.3f}s".format(time() - tic))

y_pred_test = svr.predict(x_test_std)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print ("RMSE: %.4f" % rmse)
print ("r2(score): %.4f" % r2_score(y_test, y_pred_test))

print("Training MLPRegressor...")
tic = time()
est = make_pipeline(QuantileTransformer(),
                    MLPRegressor(hidden_layer_sizes=(50, 50),
                                 learning_rate_init=0.01,
                                 early_stopping=True))
est.fit(x_train_std, y_train)
print("done in {:.3f}s".format(time() - tic))

y_pred_test = est.predict(x_test_std)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print ("RMSE: %.4f" % rmse)
print("Test R2 score: {:.4f}".format(r2_score(y_test, y_pred_test )))

kr_cv = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                  param_grid={"alpha": [100,10,1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 3)})
tic = time()
kr_cv.fit(x_train_std, y_train)
print("done in {:.3f}s".format(time() - tic))

y_test_pred_cv = kr_cv.predict(x_test_std)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred_cv))
print ("RMSE: %.4f" % rmse)
print ("r2(score): %.4f" % r2_score(y_test, y_test_pred_cv))

GBR = GradientBoostingRegressor()
paramters = {"n_estimators":[50,100,125,150,200],
             "max_depth"   :[3,5,7],
             "loss"        :["ls", "lad", "huber", "quantile"]
            }
grid = GridSearchCV(estimator=GBR,
                    param_grid=paramters,
                    scoring="r2",
                    cv=5,
                    n_jobs=-1)

tic = time()
grid.fit(x_train_std, y_train)
print("done in {:.3f}s".format(time() - tic))

y_pred_gbr = grid.predict(x_test_std)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred_gbr))
print ("RMSE: %.4f" % rmse)
print("Variance score: {}".format(r2_score(y_test, y_pred_gbr)))
model = grid.best_estimator_
print(model)
for coef in zip(x_train.columns, model.feature_importances_):
    print(coef)

XECO2_1516= X_1516[['Campus','Multifamily HR (10+)','Multifamily LR (1-4)','Multifamily MR (5-9)','NonResidential','Nonresidential COS','Nonresidential WA','SPS-District K-12','NumberofBuildings','YearBuilt','NumberofFloors','PropertyGFATotal','Longitud','Latitud','data_EmissionsCO2','DataYear']]

X= XECO2_1516.iloc[:,0:-2]
y= XECO2_1516.iloc[:,-2]
X.shape

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y,
                                test_size=0.3, random_state=42) # 30% des données dans le jeu de test

std_scale = preprocessing.StandardScaler().fit(x_train)
x_train_std = std_scale.transform(x_train)
x_test_std = std_scale.transform(x_test)

lr = linear_model.LinearRegression()
tic = time()
lr.fit(x_train_std, y_train)
print("done in {:.3f}s".format(time() - tic))

pred_train_lr= lr.predict(x_train_std)
print(np.sqrt(mean_squared_error(y_train,pred_train_lr)))
print(r2_score(y_train, pred_train_lr))

pred_test_lr= lr.predict(x_test_std)
print(np.sqrt(mean_squared_error(y_test,pred_test_lr)))
print(r2_score(y_test, pred_test_lr))

rr = linear_model.Ridge(alpha=0.1, tol=0.01)
tic = time()
rr.fit(x_train_std, y_train)
print("done in {:.3f}s".format(time() - tic))

pred_train_rr= rr.predict(x_train_std)
print(np.sqrt(mean_squared_error(y_train,pred_train_rr)))
print(r2_score(y_train, pred_train_rr))

pred_test_rr= rr.predict(x_test_std)
print(np.sqrt(mean_squared_error(y_test,pred_test_rr)))
print(r2_score(y_test, pred_test_rr))

rr_ECO2 = linear_model.Ridge()
parametr= {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 30,50,100]}
ridge_reco2=GridSearchCV(rr_ECO2,parametr,scoring='neg_mean_squared_error', cv=5)

tic = time()
ridge_reco2.fit(x_train_std, y_train)
print("done in {:.3f}s".format(time() - tic))

print(ridge_reco2.best_params_)
print(ridge_reco2.best_score_)

pred_train_rr_eco2= ridge_reco2.predict(x_train_std)
print(np.sqrt(mean_squared_error(y_train,pred_train_rr)))
print(r2_score(y_train, pred_train_rr))

pred_test_rr_eco2= ridge_reco2.predict(x_test_std)
print(np.sqrt(mean_squared_error(y_test,pred_test_rr)))
print(r2_score(y_test, pred_test_rr))

ker_ridge_cv = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 2)})

tic = time()
ker_ridge_cv.fit(x_train_std, y_train)
print("done in {:.3f}s".format(time() - tic))

y_test_pred_cv_eco2 = ker_ridge_cv.predict(x_test_std)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred_cv_eco2))
print ("RMSE: %.4f" % rmse)
print ("r2(score): %.4f" % r2_score(y_test, y_test_pred_cv_eco2))

svr = GridSearchCV(SVR(kernel='rbf', gamma=0.01),
                   param_grid={"C": [1e0, 1e1, 1e2],
                               "gamma": np.logspace(-2, 2, 3)})


tic = time()
svr.fit(x_train_std, y_train)
print("done in {:.3f}s".format(time() - tic))

y_pred_svr_co2 = svr.predict(x_test_std)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_svr_co2))
print ("RMSE: %.4f" % rmse)
print ("r2(score): %.4f" % r2_score(y_test, y_pred_svr_co2))

print("Training MLPRegressor...")
tic = time()
est = make_pipeline(QuantileTransformer(),
                    MLPRegressor(hidden_layer_sizes=(50, 50),
                                 learning_rate_init=0.01,
                                 early_stopping=True))
est.fit(x_train_std, y_train)
print("done in {:.3f}s".format(time() - tic))

y_pred_test = est.predict(x_test_std)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print ("RMSE: %.4f" % rmse)
print("Test R2 score: {:.4f}".format(r2_score(y_test, y_pred_test )))

GBR_ECO2 = GradientBoostingRegressor()
paramters = {"n_estimators":[50,100,125,150,200],
             "max_depth"   :[3,5,7],
             "loss"        :["ls", "lad", "huber", "quantile"]
            }
grid_ECO2 = GridSearchCV(estimator=GBR_ECO2,
                    param_grid=paramters,
                    scoring="r2",
                    cv=5,
                    n_jobs=-1)
tic = time()
grid_ECO2.fit(x_train_std, y_train)
print("done in {:.3f}s".format(time() - tic))

y_pred_gbr_ECO2 = grid_ECO2.predict(x_test_std)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred_gbr_ECO2))
print ("RMSE: %.4f" % rmse)
print("Variance score: {}".format(r2_score(y_test, y_pred_gbr_ECO2)))
model = grid_ECO2.best_estimator_
print(model)
for coef in zip(x_train.columns, model.feature_importances_):
    print(coef)

#Gradient Boosted Regression Trees avec cross-validation Model: Test avec une valeur RMSE = 282.95 et r2(score)= 70.19 %.