#Machine Learning for Public Policy: HW1
#Charity King
#PA3: Machine Learning Pipeline
#May 3, 2016
#csking1@uchicago.edu

import pandas as pd
import explore_clean as exp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


MODELS = ['KNN','RF','LR','GB','NB','DT']

def params():
	clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(n_neighbors=3) 
            }

	grid = {'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }

def go(training, testing):
    #########Initial phase: Reading, imputing data
	test = exp.read_data(testing)
    df = exp.read_data(training)
	exp.data_summary(df)
	exp.graph_data(df)
	exp.impute_data(df, mean=False, median=True)

def main_pipeline():
	for model in MODELS:
		print ("Model:", model)



def clf_loop():
	pass
    # for n in range(1, 2):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #     for index,clf in enumerate([clfs[x] for x in models_to_run]):
    #         print models_to_run[index]
    #         parameter_values = grid[models_to_run[index]]
    #         for p in ParameterGrid(parameter_values):
    #             try:
    #                 clf.set_params(**p)
    #                 print clf
    #                 y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
    #                 #threshold = np.sort(y_pred_probs)[::-1][int(.05*len(y_pred_probs))]
    #                 #print threshold
    #                 print precision_at_k(y_test,y_pred_probs,.05)
    #                 #plot_precision_recall_n(y_test,y_pred_probs,clf)
    #             except IndexError, e:
    #                 print 'Error:',e
    #                 continue


if __name__ == '__main__':
	training = "cs-training.csv"
	testing = "cs-test.csv"

go(training, testing)
main_pipeline()

 