from datetime import datetime
import itertools

import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer, Imputer, LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, SelectPercentile
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold

FILENAME = "D:/datasets/churn/tourn_1_calibration_csv/tourn_1_calibration_csv.csv"
SAVE_FILE = "D:/datasets/churn/tourn_1_calibration_csv/preprocessed8.csv"

data = pd.read_csv(FILENAME, encoding="utf")
X = data.ix[:, -1] # customer ID feature

all_cols = data.columns

# categorical columns
cat_cols = ["actvsubs", "adults", "age1", "age2", "area", "asl_flag", "car_buy", "cartype", "children", \
	"churn", "crclscod", "creditcd", "crtcount", "csa", "Customer_ID", "div_type", "dualband", "dwllsize", \
	"dwlltype", "educ1", "ethnic", "forgntvl", "hnd_price", "HHstatin", "hnd_webcap", "income", "infobase", \
	"kid0_2", "kid3_5", "kid6_10", "kid11_15", "kid16_17", "last_swap", "lor", "mailflag", "mailordr", "mailresp", \
	"marital", "models", "mtrcycle", "new_cell", "numbcars", "occu1", "ownrent", "pcowner", "phones", "pre_hnd_price", \
	"prizm_social_one", "proptype", "ref_qty", "refurb_new", "rv", "solflag", "tot_ret","tot_acpt", "truck", "uniqsubs", "wrkwoman"]

# *** Replace kid0-15 columns with preteen feature ***
preteen = data[["kid0_2","kid3_5","kid6_10"]].apply(lambda x: "Y" if (x.ix[0] == "Y" or x.ix[1] == "Y" or x.ix[2] == "Y") else "N", axis=1)
data ["preteen"] = preteen
data.drop(["kid0_2","kid3_5","kid6_10"], axis=1, inplace=True)
# ************************************************
# *** Replace truck, motorcycle, rv columns with vehicle feature ***
vehicle = data[["rv","mtrcycle","truck"]].apply(lambda x: 1 if (x.ix[0] == 1 or x.ix[1] == 1 or x.ix[2] == 1) else 0, axis=1)
data ["vehicle"] = vehicle
data.drop(["rv","mtrcycle","truck"], axis=1, inplace=True)
# ************************************************

# continuous columns
cont_cols = set(all_cols) - set(cat_cols)
print "length of continuous: ", len(cont_cols)
print "continuous: ", cont_cols

N = data.shape[0]

cat_cols.append("preteen")
cat_cols.append("vehicle")

cat_cols.remove("kid0_2")
cat_cols.remove("kid3_5")
cat_cols.remove("kid6_10")
cat_cols.remove("rv")
cat_cols.remove("mtrcycle")
cat_cols.remove("truck")

# categorical variable
print
print "length of categorical: ", len(cat_cols)
print "categorical: ", cat_cols
print

# Handle Categorical columns
impfreq = Imputer(strategy="most_frequent", axis=1)
lb = LabelEncoder()

all_columns = ["Customer_ID",]
for column in cat_cols:	
	if column == "Customer_ID" or column == "churn":
		continue
	
	# *** Remove columns with > 95% missing values ***
	missing = pd.isnull(data[column])
	missing = missing[missing]

	if (missing.shape[0] > 0) and (float(float(missing.shape[0]) / float(N)) > 0.95):
		print 
		print column, missing.shape
		continue
	# ************************************************

	all_columns.append(column)
	datatype = data[column].dtype
	if datatype != "int64":	
		col = lb.fit_transform(data[column])
		col = impfreq.fit_transform(col)
		X = np.column_stack((X, col.T))
	else:
		col = impfreq.fit_transform(data[column])
		X = np.column_stack((X, col.T))

anova_filter = SelectPercentile(f_classif, percentile=80)
X_cat = anova_filter.fit_transform(X[:,1:], data["churn"])

cols = [all_columns[i+1] for i in anova_filter.get_support(indices=True)]
print 'anova columns: ', cols
all_columns = ["Customer_ID"]
all_columns = all_columns + cols
print "all columns: ", all_columns

X_new = X[:,0]
X = np.column_stack((X_new, X_cat))

# Handle Continuous columns
impmedian = Imputer(strategy="median", axis=1)

for column in list(cont_cols):
	# *** Remove columns with > 95% missing values ***
	missing = pd.isnull(data[column])
	missing = missing[missing]

	if (missing.shape[0] > 0) and (float(float(missing.shape[0]) / float(N)) > 0.95):
		print 
		print column, missing.shape
		continue
	# *************************************************

	all_columns.append(column)
	
	col = impmedian.fit_transform(data[column])
	col = Normalizer().fit_transform(col)

	X = np.column_stack((X, col.T))

print X.shape, len(all_columns)

# use decision forest predictions as new features

dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
kf = KFold(data.shape[0], n_folds=3, random_state=42)
pred_probas = np.zeros((data.shape[0], 2))

X1, y1 = X[:, 1:], np.array(data["churn"])
print X1.shape, y1.shape
print 'X and Y'
print y1

for train_index, test_index in kf:
	X_train, X_test = X1[train_index], X1[test_index]
	y_train, y_test = y1[train_index], y1[test_index]
	print X_train.shape, X_test.shape, y_train.shape, y_test.shape

	dt.fit(X_train, y_train)
	preds = dt.predict_proba(X_test)
	print preds.shape, pred_probas[test_index].shape
	pred_probas[test_index] = preds

X = np.column_stack((X, pred_probas))
X = np.column_stack((X, data["churn"]))

all_columns.append("decisiontree_nochurn")
all_columns.append("decisiontree_churn")
all_columns.append("churn")

# Finally, save to disk
pd.DataFrame(X).to_csv(SAVE_FILE, index=False, header=all_columns)