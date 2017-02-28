###this file is aimed to calculate the accuracy based on the voting classifier
import numpy as np
import pandas as pd
from scipy import stats
import random
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
##loading and processing data
##logistic regression, random forest classifier, gaussianNB
def main_function():
	importance=['price','avg_imagesize_x','word_count','avg_luminance_x','avg_brightness_x','manager count','description_sentiment','img_quantity_x','unique_count','bedrooms','bathrooms','No Fee',\
	'dist_count','Doorman','Laundry In Building','Elevator','Fitness Center','Reduced Fee','Exclusive','Cats Allowed','Dogs Allowed','Furnished',\
	'Common Outdoor Space','Laundry In Unit','Private Outdoor Space','Parking Space','Short Term Allowed','By Owner','Sublet / Lease-Break',\
	'Storage Facility']
	processed_data=pd.read_json("../data/processed_train.json")
	img=pd.read_csv("../data/image_stats-fixed.csv",index_col=0)
	processed_data=processed_data.merge(img,how="left",on="listing_id")
	processed_data=processed_data.fillna(0)
	print "data processed"

	train_data=processed_data.sample(n=processed_data.shape[0]*8/10)
	test_data=processed_data.drop(train_data.index)

	train=train_data.drop(['building_id','created','description','display_address','manager_id','longitude','latitude','listing_id','photos','street_address','features'],axis=1)
	test=test_data.drop(['building_id','created','description','display_address','manager_id','longitude','latitude','listing_id','photos','street_address','features'],axis=1)
	ans=[['Features','Train','Test']]
	
	y_train=train.loc[:,'interest_level']
	x_train=train.drop('interest_level',axis=1)
	y_test=test.loc[:,'interest_level']
	x_test=test.drop('interest_level',axis=1)
	
	print "-"*150+"\ndata created"

	res=addnew(x_train,y_train)
	print res

	

def addnew(x_train,y_train):
	clf = [LogisticRegression(C=100),RandomForestClassifier(n_estimators=200),GaussianNB(),DecisionTreeClassifier(max_depth=4),GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,max_depth=4),AdaBoostClassifier(n_estimators=100)]
	score=[]
	print "-"*50+"\nmodel created"
	for i in range(6):
		estimator=[('lr', clf[0]), ('rf', clf[1]), ('gnb', clf[2]),('dt',clf[3]),('gb',clf[4]),('rf2',clf[1]),('rf3',clf[1]),('gb2',clf[4])]
		estimator.append(('new',clf[i]))
		print "-"*150+"\nestimator created"
		# if i==0:
		# 	params = {'lr__C': [1, 100], 'rf__n_estimators': [20, 200],'new__C': [1, 100]}
		# elif i==1:
		# 	params = {'lr__C': [1, 100], 'rf__n_estimators': [20, 200],'new__n_estimators': [20, 200]}
		print "-"*150+"\nparams created"
		eclf = VotingClassifier(estimators=estimator, voting='soft')
		score.append(np.mean(cross_val_score(eclf, x_train, y_train,n_jobs=2)))
		print "-"*150+"\nscore created"
		print "Score for model ", i, "is ", score[-1]

	return score

main_function()
