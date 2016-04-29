# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 21:44:33 2016

@author: Alicia Jin
"""

import numpy as np
import pandas as pd 
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn import neighbors
from sklearn import cross_validation, linear_model
from sklearn.cross_validation import KFold

train = pd.read_csv('training2.csv')
valid = pd.read_csv('validation2.csv')

#### preprocessing
train['transdate'] = pd.to_datetime(train['transdate'])
train['transdate_previous'] = pd.to_datetime(train['transdate_previous'])
valid['transdate'] = pd.to_datetime(valid['transdate'])
valid['transdate_previous'] = pd.to_datetime(valid['transdate_previous'])
	
train['builtyear'] = map(lambda x: 2016 - int(x if x==x else 2016), train['builtyear'])
valid['builtyear'] = map(lambda x: 2016 - int(x if x==x else 2016), valid['builtyear'])

train['viewtypeid'].fillna(0, inplace = True)
valid['viewtypeid'].fillna(0, inplace = True)

#####
train = train.drop(['propertyid', 'usecode'], axis = 1) 
valid = valid.drop(['propertyid', 'usecode'], axis = 1) 

##### processing validation set missing data, fill with mean value
train.isnull().sum()
valid.isnull().sum()
avg_bathroomcnt = np.mean(valid['bathroomcnt'])
avg_bedroomcnt = np.mean(valid['bedroomcnt'])
avg_finishedsquarefeet = np.mean(valid['finishedsquarefeet'])
avg_storycnt = np.mean(valid['storycnt'])

valid['bathroomcnt'].fillna(avg_bathroomcnt,inplace = True)
valid['bedroomcnt'].fillna(avg_bedroomcnt,inplace = True)
valid['finishedsquarefeet'].fillna(avg_finishedsquarefeet,inplace = True)
valid['storycnt'].fillna(avg_storycnt,inplace = True)


## outlier, extreme data
train = train.drop(1098) 
train = train.drop(34) 

### adjust features and create new features
train['transyr_previous'] = train['transdate_previous'].dt.year
valid['transyr_previous'] = valid['transdate_previous'].dt.year

## inflation rate during 1988-2005 year
inflation = {'1988': 0.65, '1989':0.58, '1990': 0.49, '1991': 0.43, '1992': 0.39, '1993': 0.35, 
             '1994': 0.32, '1995': 0.28, '1996': 0.25, '1997': 0.22, '1998': 0.20, '1999': 0.17,
             '2000': 0.13, '2001': 0.10, '2002': 0.09, '2003': 0.06, '2004': 0.06, '2005': 0.00}
## get discount value
train['transvalue_discount'] = map(lambda v,yr: v+v*inflation[str(int(yr))] if v==v else v, train['transvalue_previous'], train['transyr_previous'] )
valid['transvalue_discount'] = map(lambda v,yr: v+v*inflation[str(int(yr))] if v==v else v, valid['transvalue_previous'], valid['transyr_previous'] )

inflation_house = {'1988': 1.00, '1989':1.09, '1990': 1.40, '1991': 1.54, '1992': 1.58, '1993': 1.613, 
             '1994': 1.665, '1995': 1.690, '1996': 1.757, '1997': 1.824, '1998': 2.006, '1999': 2.182,
             '2000': 2.382, '2001': 2.525, '2002': 2.634, '2003': 2.772, '2004': 2.969, '2005': 3.371}

inflation_house2 = inflation_house
for x in inflation_house:
    inflation_house2[x] = 3.371/inflation_house[x]
## get discount value
train['transvalue_discount2'] = map(lambda v,yr: v+v*inflation_house2[str(int(yr))] if v==v else v, train['transvalue_previous'], train['transyr_previous'] )
valid['transvalue_discount2'] = map(lambda v,yr: v+v*inflation_house2[str(int(yr))] if v==v else v, valid['transvalue_previous'], valid['transyr_previous'] )

## rough estimator for area of each bedroom
train['size_bedroom'] = map(lambda x,y: x*1.0/y, train['finishedsquarefeet'], train['bedroomcnt'])
valid['size_bedroom'] = map(lambda x,y: x*1.0/y, valid['finishedsquarefeet'], valid['bedroomcnt'])

train['transmonth'] = train['transdate'].dt.month
valid['transmonth'] = valid['transdate'].dt.month

train['bed/bath'] = map(lambda x,y: 1.0*x/y , train['bedroomcnt'], train['bathroomcnt'])
valid['bed/bath'] = map(lambda x,y: 1.0*x/y , valid['bedroomcnt'], valid['bathroomcnt'])

train['bed*bath'] = map(lambda x,y: 1.0*x*y , train['bedroomcnt'], train['bathroomcnt'])
valid['bed*bath'] = map(lambda x,y: 1.0*x*y , valid['bedroomcnt'], valid['bathroomcnt'])



''' validation set part 1: model build with previous transaction information''' 

''' lasso model '''
########## select features
train.columns.values.tolist()
select_col = ['transvalue', 'bathroomcnt', 'bedroomcnt', 'builtyear', 'finishedsquarefeet',
              'lotsizesquarefeet', 'storycnt', 'latitude', 'longitude', 'censustract', 
              'viewtypeid', 'avgprice*sqr', 'transyr_previous', 'transvalue_discount2', 
              'size_bedroom', 'transmonth', 'bed/bath', 'bed*bath']
res = get_features(select_col)
train_X, train_y, valid_X, valid_y, fea_names = res[0], res[1], res[2], res[3], res[4]

lasso_plot_alpha2(np.logspace(2.5, 5, 20, dtype='int'))
best_alpha = np.logspace(2.5, 5, 20, dtype='int')[8] ## 3569
lasso = linear_model.Lasso(alpha = best_alpha)
CV(lasso, 5)
zip(fea_names, [int(x) for x in lasso.coef_])

## entire train, valid set
lasso = linear_model.Lasso(alpha = best_alpha)
lasso.fit(train_X,train_y)
y_hat_valid = lasso.predict(valid_X)
report(y_hat_valid, valid_y)
y_hat_train = lasso.predict(train_X)
report(y_hat_train, train_y )


''' SVR model '''
select_col = ['transvalue', 'bedroomcnt',  'finishedsquarefeet',
              'lotsizesquarefeet', 'censustract', 
              'avgprice*sqr', 'transyr_previous', 'transvalue_discount2', 
              'size_bedroom', ]
res = get_features(select_col)
train_X, train_y, valid_X, valid_y, fea_names = res[0], res[1], res[2], res[3], res[4]
svr_lin = SVR(kernel='linear', C = 5000)
CV(svr_lin, 5)
### entire train. valid set
svr_lin.fit(train_X, train_y)
y_hat_valid = svr_lin.predict(valid_X)
report(y_hat_valid, valid_y)
y_hat_train = svr_lin.predict(train_X)
report(y_hat_train, train_y )


''' knn model '''
select_col = ['transvalue', 
              'avgprice*sqr', 'transyr_previous', 'transvalue_discount2', 
               ]          
res = get_features(select_col)
train_X, train_y, valid_X, valid_y, fea_names = res[0], res[1], res[2], res[3], res[4]
best_k = knn_get_k(range(1,8))
knn = neighbors.KNeighborsClassifier(n_neighbors = best_k)
CV(knn, 5)
### entire train. valid set
knn.fit(train_X, train_y)
y_hat_valid = knn.predict(valid_X)
report(y_hat_valid, valid_y)
y_hat_train = knn.predict(train_X)
report(y_hat_train, train_y )


''' validation set part 2: valid dataset without previous transaction information'''
ind_null = pd.isnull(valid).any(1).nonzero()[0]
valid_withnull = valid.loc[ind_null]

select_col = ['transvalue', 'bedroomcnt', 'builtyear', 'finishedsquarefeet',
              'lotsizesquarefeet', 'censustract', 
              'avgprice*sqr',  
              'size_bedroom', 'transmonth', 'bed/bath', 'bed*bath']

res = get_features(select_col)
train_X, train_y, valid_X, valid_y, fea_names = res[0], res[1], res[2], res[3], res[4]
#### do same model building stepsas above


#### function built here
def get_features(select_col):
    train2 = train[select_col].dropna()
    train2 = train2.reset_index(drop=True)
    valid2 = valid[select_col].dropna()
    valid2 = valid2.reset_index(drop=True)
    
    train_X = train2.drop(['transvalue'], axis = 1)
    valid_X = valid2.drop(['transvalue'], axis = 1)
    train_y = train2['transvalue']
    valid_y = valid2['transvalue']
    fea_names = train_X.columns.values.tolist()
    ### scale
    std_X = StandardScaler().fit(train_X)
    train_X = pd.DataFrame( std_X.transform(train_X) )
    valid_X = pd.DataFrame( std_X.transform(valid_X) )
    return [train_X, train_y, valid_X, valid_y, fea_names]

def lasso_plot_alpha2(list_a):
    mae_list, percent5_list, percent10_list, percent20_list  = [], [], [], []
    for a in list_a:
        lasso = linear_model.Lasso(alpha = a)
        res = CV(lasso, 5)
        mae_list.append(res[0])
        percent5_list.append(res[1])
        percent10_list.append(res[2])
        percent20_list.append(res[3])
    plt.figure(1)
    plt.subplot(121)
    plt.plot(mae_list)
    plt.title('mae ~ alpha')
    plt.subplot(122)
    plt.plot(percent5_list)
    plt.plot(percent10_list)
    plt.plot(percent20_list)
    plt.title('20%, 10%, 5% ~ alpha')


def knn_get_k(range_k):
    score_list = []
    best_score, best_k = 0, 0
    for k in range_k:
        knn = neighbors.KNeighborsClassifier(n_neighbors = k)
        scores = cross_validation.cross_val_score(knn, train_X, train_y, cv=5)
        avg = np.mean(scores)
        score_list.append(avg)
        if avg > best_score:
            best_score, best_k = avg, k
    print 'best k is: %d'%best_k
    plt.plot(score_list,'bo--')
    return best_k



def CV(the_model, fold):
    kfold = KFold(n=len(train_y), n_folds=fold, shuffle = True)
    mae_list, percent5_list, percent10_list, percent20_list  = [], [], [], []
    for train_ind, vali_ind in kfold:
        X_train, X_vali = train_X.loc[train_ind], train_X.loc[vali_ind]
        y_train, y_vali = train_y.loc[train_ind], train_y.loc[vali_ind]
        model = the_model
        pred = model.fit(X_train, y_train).predict(X_vali)
        mae = np.median(np.abs(pred-y_vali))
        mae_list.append(mae)
        variation = np.abs(pred - y_vali) / y_vali
        percent5 = sum([1 for i in variation if i <=0.05]) * 1.0 / len(y_vali)
        percent5_list.append(percent5)
        percent10 = sum([1 for i in variation if i <=0.1]) * 1.0 / len(y_vali)
        percent10_list.append(percent10)
        percent20 = sum([1 for i in variation if i <=0.2]) * 1.0 / len(y_vali)
        percent20_list.append(percent20)
    mae_avg = np.mean(mae_list)
    percent5_avg = np.mean(percent5_list)
    percent10_avg = np.mean(percent10_list)
    percent20_avg = np.mean(percent20_list)
    return [format(mae_avg, '.3f'), format(percent5_avg, '.3f'), format(percent10_avg, '.3f'), format(percent20_avg, '.3f')]
    
    
def report(pred, y_vali):
    mae = np.median(np.abs(pred-y_vali))
    variation = np.abs(pred - y_vali) / y_vali
    percent5 = sum([1 for i in variation if i <=0.05]) * 1.0 / len(y_vali)
    percent10 = sum([1 for i in variation if i <=0.1]) * 1.0 / len(y_vali)
    percent20 = sum([1 for i in variation if i <=0.2]) * 1.0 / len(y_vali)
    return [format(mae, '.3f'), format(percent5, '.3f'), format(percent10, '.3f'), format(percent20, '.3f')]
    #return [mae, percent5, percent10, percent20]        


#### overview plot 
def scatter(key1, key2):
    data = train[[key1, key2]].dropna()
    plt.plot(data[key1], data[key2], 'bo--')
    plt.xlabel(key1)
    plt.ylabel(key2)
    plt.gcf().autofmt_xdate()


#### plot all, ooptional
'''
index1 = [241,242,243,244,245,246,247,248]
index2 = [241,242,243,244,245]

features = train.columns.values.tolist()
features.remove('transvalue')
plt.figure(1)
for i in range(len(index1)):
    plt.subplot(index1[i])
    scatter(features[i], 'transvalue')

plt.figure(2)
for i in range(len(index2)):
    j = len(index1)+i
    plt.subplot(index2[i])
    scatter(features[j], 'transvalue')
'''
'''
index1 = [241,242,243,244,245,246,247,248]
index2 = [241,242,243,244,245]

yfea = 'transvalue'
plt.figure(1)
for i in range(len(index1)):
    plt.subplot(index1[i])
    data = train[ [features[i], yfea] ].dropna()
    plt.plot(data[features[i]], data[yfea], 'bo')
    plt.ylabel(yfea)
    plt.xlabel(features[i])
    plt.gcf().autofmt_xdate()

plt.figure(2)
for i in range(len(index2)):
    j = len(index1)+i
    plt.subplot(index2[i])
    data = train[ [features[j], yfea] ].dropna()
    plt.plot(data[features[j]], data[yfea], 'bo')
    plt.ylabel(yfea)
    plt.xlabel(features[j])
    plt.gcf().autofmt_xdate()

####  plot geographic locations to trans value.
plt.figure(1)
plt.scatter(train['longitude'], train['latitude'], c = np.log(train['transvalue']), s = 40)
plt.colorbar()
plt.xlabel('longitude')
plt.ylabel('latitude')
'''


