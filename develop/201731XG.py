import numpy as np
import pandas as pd
from scipy import stats
import random
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
def main_function():
	importance=['price','avg_imagesize_x','word_count','avg_luminance_x','avg_brightness_x','manager count','description_sentiment','img_quantity_x','unique_count','bedrooms','bathrooms','No Fee',\
	'dist_count','Doorman','Laundry In Building','Elevator','Fitness Center','Reduced Fee','Exclusive','Cats Allowed','Dogs Allowed','Furnished',\
	'Common Outdoor Space','Laundry In Unit','Private Outdoor Space','Parking Space','Short Term Allowed','By Owner','Sublet / Lease-Break',\
	'Storage Facility']
	processed_data=pd.read_json("../data/processed_train.json")
	img=pd.read_csv("../data/image_stats-fixed.csv",index_col=0)
	processed_data=processed_data.merge(img,how="left",on="listing_id")
	processed_data=processed_data.fillna(0)
	train_data=processed_data.sample(n=processed_data.shape[0]*7/10)
	test_data=processed_data.drop(train_data.index)
	train=train_data.drop(['building_id','created','description','display_address','longitude','latitude','manager_id','listing_id','photos','street_address','features'],axis=1)
	test=test_data.drop(['building_id','created','description','display_address','longitude','latitude','manager_id','listing_id','photos','street_address','features'],axis=1)
	ans=[['Features','Train on meta','Test on meta','Train with all','Test with all']]
	y_train=train.loc[:,'interest_level']
	x_train=train.drop('interest_level',axis=1).loc[:,importance[:16]]
	y_test=test.loc[:,'interest_level']
	x_test=test.drop('interest_level',axis=1).loc[:,importance[:16]]
	y_train_copy=y_train.copy()
	y_test_copy=y_test.copy()
	diction={'low':0,'medium':1,'high':2}
	y_train1=map(lambda x: diction[x],y_train)
	y_test1=map(lambda x: diction[x],y_test)
	y_train=pd.Series(y_train1,index=y_train.index)
	y_test=pd.Series(y_test1,index=y_test.index)
# this is for max_depth and min_child_weight
	param_test1 = {
	 'max_depth':range(3,10,10),
	 'min_child_weight':range(1,6,10)
	}
	gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=200, max_depth=5,
	min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
	objective= 'multi:softmax', nthread=3, scale_pos_weight=1, seed=27), 
	param_grid = param_test1, scoring='roc_auc',n_jobs=12,iid=False, cv=5,num_class=3)
	gsearch1.fit(x_train,y_train)
	gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
# this part is for gamma
	# param_test3 = {
	#  'gamma':[i/10.0 for i in range(0,5,10)]
	# }
	# gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
	# min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
	# objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=27), 
	# param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5,num_class=3)
	# gsearch3.fit(train[predictors],train[target])
	# gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

# this part is for subsample and colsample
	# param_test4 = {
 	#'subsample':[i/10.0 for i in range(6,10,10)],
	# 'colsample_bytree':[i/10.0 for i in range(6,10,10)]
	# }
	# gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
	# min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
	# objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=27), 
	# param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5,num_class=3)	
	# gsearch4.fit(train[predictors],train[target])
	# gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

# this part is for alpha
	# param_test6 = {
	#  'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
	# }
	# gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
	# min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
	# objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=27), 
	# param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5,num_class=3)
	# gsearch6.fit(train[predictors],train[target])
	# gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
main_function()
