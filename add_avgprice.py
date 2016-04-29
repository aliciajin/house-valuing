# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:09:14 2016

@author: Alicia Jin
"""

'''
new api
>> pip install geopy
'''
import numpy as np
import pandas as pd 
from geopy.geocoders import Nominatim
from sklearn import neighbors

train = pd.read_csv('training.csv')
valid = pd.read_csv('validation.csv')

train['latitude'] = train['latitude']/1000000.0
valid['latitude'] = valid['latitude']/1000000.0
train['longitude'] = train['longitude']/1000000.0
valid['longitude'] = valid['longitude']/1000000.0

zipcode_train, i = [], 0
geolocator = Nominatim()

for x,y in zip(train['latitude'].loc[392:], train['longitude'].loc[392:]):
    location = geolocator.reverse(str(x)+', '+str(y)).raw
    print i
    if u'address' in location:
        if u'postcode' in location[u'address']:
            zcode = int( location[u'address'][u'postcode'][:5] )
    else:
        zcode = 'tbd'
        print x,y
    i+=1
    zipcode_train.append(zcode)
    
################################
############
for x,y in zip(valid['latitude'].loc[2276:], valid['longitude'].loc[2276:]):
    location = geolocator.reverse(str(x)+', '+str(y)).raw
    print j
    if u'address' in location:
        if u'postcode' in location[u'address']:
            zcode = int( location[u'address'][u'postcode'][:5] )
    else:
        zcode = 'tbd'
        print x,y
    j+=1
    zipcode_valid.append(zcode)
 
train['zipcode'] = pd.DataFrame(zipcode_train)
valid['zipcode'] = pd.DataFrame(zipcode_valid)

### get average price  ####
avgprice = pd.read_csv('avg_value_2015.csv')
avgprice['fake'] = [0]*len(avgprice)

knn = neighbors.KNeighborsClassifier(n_neighbors = 1)
p_X = avgprice[['RegionName', 'fake']]
p_y = avgprice['avgprice']
knn.fit(p_X, p_y)

train['fake'] = [0]*len(train)
valid['fake'] = [0]*len(valid)
pred_avgp = knn.predict(train[['zipcode', 'fake']])
pred_avgp2 = knn.predict(valid[['zipcode', 'fake']])
train['avgprice'] = pd.DataFrame(pred_avgp)
valid['avgprice'] = pd.DataFrame(pred_avgp2)

train['avgprice*sqr'] = map(lambda x,y: x*y, train['avgprice'], train['finishedsquarefeet'])
valid['avgprice*sqr'] = map(lambda x,y: x*y, valid['avgprice'], valid['finishedsquarefeet'])

scatter('avgprice*sqr', 'transvalue')

train = train.drop(['fake'], axis = 1)
valid = valid.drop(['fake'], axis = 1)
#####

train.to_csv('training2.csv')
valid.to_csv('validation2.csv')






