from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split, KFold
from sklearn.calibration import calibration_curve
from sklearn import grid_search
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score, roc_curve, confusion_matrix)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ensemble import EnsembleClassifier

# FILENAME1 = "D:/datasets/churn/tourn_1_calibration_csv/tourn_1_calibration_csv.csv"
# FILENAME2 = "D:/datasets/churn/tourn_1_calibration_csv/preprocessed2.csv" # Normalized/encoded Features
# FILENAME3 = "D:/datasets/churn/tourn_1_calibration_csv/preprocessed3.csv" # Modified Features:
# FILENAME4 = "D:/datasets/churn/tourn_1_calibration_csv/preprocessed4.csv" # Removed range Features
# FILENAME5 = "D:/datasets/churn/tourn_1_calibration_csv/preprocessed5.csv" # Using Feature selection for regression
# FILENAME6 = "D:/datasets/churn/tourn_1_calibration_csv/preprocessed6.csv" # Using Feature selection for classification
# FILENAME7 = "D:/datasets/churn/tourn_1_calibration_csv/preprocessed7.csv" # Combination of range features and classification feature selection
FILENAME8 = "D:/datasets/churn/tourn_1_calibration_csv/preprocessed8.csv" # With decision tree predictions as new features

metrics = np.zeros((4, 4))

def train_rf(filename, color, name):
	'''Train on Random Forest Classifier'''
	# Read data
	data2 = pd.read_csv(filename, encoding="utf") 
	X = data2.ix[:, 1:-1]
	y = data2.ix[:, -1]

	# Split into train, validation and test
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

	# Define models
	clf = RandomForestClassifier(n_estimators=150, random_state=42, criterion="entropy", max_depth=None)
	
	# Fit model
	# parameters = {'n_estimators':[25, 50, 100], 'criterion':('gini', 'entropy'), 'max_depth': [None, 5, 10]}
	# clf = grid_search.GridSearchCV(clf3, parameters)

	t0 = time()
	clf.fit(X_train, y_train)
	pred_probas = clf.predict_proba(X_val)

	predictions = clf.predict(X_val)
	
	print "Score", clf.score(X_val, y_val)
	
	importances = clf.feature_importances_
	indices = np.argsort(importances)[::-1]
	
	# Metrics & Plotting
	metrics[0, 0] = precision_score(y_val, predictions)
	metrics[0, 1] = recall_score(y_val, predictions)
	metrics[0, 2] = f1_score(y_val, predictions)
	metrics[0, 3] = time() - t0

	fpr_rf, tpr_rf, _ = roc_curve(y_val, predictions)
	plt.plot(fpr_rf, tpr_rf, color=color, label=name)	
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")

	return importances, indices

def train_gbt(filename, color, name):
	'''Train on Gradient Boosted Trees Classifier'''
	# Read data
	data2 = pd.read_csv(filename, encoding="utf")
	X = data2.ix[:, 1:-1]
	y = data2.ix[:, -1]

	# Split into train, validation and test
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

	# Define model
	clf1 = GradientBoostingClassifier(learning_rate=0.05, max_depth=5, random_state=42)
	
	# Fit model
	t0 = time()
	clf1.fit(X_train, y_train)
	pred_probas = clf1.predict_proba(X_val)

	predictions = clf1.predict(X_val)
	
	print "Score", clf1.score(X_val, y_val)

	importances = clf1.feature_importances_
	indices = np.argsort(importances)[::-1]
	
	# Metrics & Plotting
	metrics[1, 0] = precision_score(y_val, predictions)
	metrics[1, 1] = recall_score(y_val, predictions)
	metrics[1, 2] = f1_score(y_val, predictions)
	metrics[1, 3] = time() - t0

	fpr_rf, tpr_rf, _ = roc_curve(y_val, predictions)
	plt.plot(fpr_rf, tpr_rf, color=color, label=name)

	return importances, indices

def sg_train(filename, color, name):
	'''Train on Stacked Generalization: Train on individual models first (RF, GBT), then use the predicted probabilities to train Logistic Regression'''
	# Read data
	data2 = pd.read_csv(filename, encoding="utf") 
	X = data2.ix[:, 1:-1]
	y = data2.ix[:, -1]

	X = np.array(X)
	y = np.array(y)

	# Define models
	clf1 = LogisticRegression(C=0.7)
	clf3 = RandomForestClassifier(n_estimators=150, random_state=42, criterion="entropy", max_depth=None)
	clf4 = GradientBoostingClassifier(learning_rate=0.05, max_depth=5, random_state=42)

	# Fit model
	kf = KFold(X.shape[0], n_folds=3, random_state=42)
	pred_probas = np.zeros((X.shape[0], 4))

	t0 = time()	
	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		clf3.fit(X_train, y_train)
		preds = clf3.predict_proba(X_test)
		pred_probas[test_index, 0:2] = preds

		clf4.fit(X_train, y_train)
		preds = clf4.predict_proba(X_test)
		pred_probas[test_index, 2:] = preds

	X_train, X_val, y_train, y_val = train_test_split(pred_probas, y, test_size=0.2, random_state=42)
	clf1.fit(X_train, y_train)
	preds = clf1.predict_proba(X_val)

	predictions = clf1.predict(X_val)
	
	print "Score", clf1.score(X_val, y_val)

	# Metrics & Plotting
	metrics[2, 0] = precision_score(y_val, predictions)
	metrics[2, 1] = recall_score(y_val, predictions)
	metrics[2, 2] = f1_score(y_val, predictions)
	metrics[2, 3] = time() - t0

	fpr_rf, tpr_rf, _ = roc_curve(y_val, predictions)
	plt.plot(fpr_rf, tpr_rf, color=color, label=name)		

def ens_train(filename, color, name):
	'''Train on Ensemble Classifier: Use Weighted Majority Voting on several classifiers '''
	data2 = pd.read_csv(filename, encoding="utf") 

	X = data2.ix[:, 1:-1]
	y = data2.ix[:, -1]

	X = np.array(X)
	y = np.array(y)

	# Define models
	clf3 = RandomForestClassifier(n_estimators=100, random_state=42)
	clf4 = GradientBoostingClassifier(learning_rate=0.05, max_depth=5, random_state=42)
	eclf = EnsembleClassifier(clfs=[clf3, clf4], voting='soft', weights=[1,2]) 
	
	# Fit Model
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
	
	t0 = time()
	eclf.fit(X_train, y_train)

	predictions = eclf.predict(X_val)
	
	# Metrics & Plotting
	metrics[3, 0] = precision_score(y_val, predictions)
	metrics[3, 1] = recall_score(y_val, predictions)
	metrics[3, 2] = f1_score(y_val, predictions)
	metrics[3, 3] = time() - t0

	fpr_rf, tpr_rf, _ = roc_curve(y_val, predictions)
	plt.plot(fpr_rf, tpr_rf, color=color, label=name)		

def main():
	print 'Training dataset for predicting customer churn:'
	
	_, _ = train_rf(FILENAME8, 'b', 'Random Forest') 
	imp, ind = train_gbt(FILENAME8, 'y', 'Random Forest') 
	sg_train(FILENAME8, 'r', 'Stacked Generalization') 
	ens_train(FILENAME8, 'g', 'Ensemble')

	print ''
	print("Feature ranking by Gradient Boosted Model:")
	for f in range(10):
	    print("%d. feature %s (%f)" % (f + 1, data2.columns[ind[f]], imp[indices[f]]))

	plt.legend()
	plt.savefig("ROC_curves_churn.png", bbox_inches='tight')

	metrics = pd.DataFrame(metrics, columns=["Precision", "Recall", "F1 score", "Time to train"])
	print metrics

	# metrics.to_csv("D:/python_files/churn/metrics.csv", encoding="utf-8", index=False)

if __name__ == '__main__':
	main()